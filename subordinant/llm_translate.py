"""LLM translation stage: Japanese text -> English, via a local Ollama.

Why this exists alongside translate.py. Sugoi v4 is a 300M NMT model that sees
one sentence and nothing else, and its two standing failures both come from
that: it renames the same character on every line, and it cannot recover a
subject the speaker dropped - which Japanese speakers do constantly, so a good
share of the utterances on a stream arrive without one. Neither is fixable by
tuning a model that has no memory to tune.

An instruction model has the memory. It is shown the last few utterances as
prior turns, so it has already seen how it rendered a name and who is being
talked about, and it holds both. That is what is being bought here, and the
price is several times the cost per call.

Where that price is affordable is the whole design:

  - Not on the live preview. Re-translating a moving tail several times a
    second is a budget of tens of milliseconds. The preview is drafted by the
    NMT model instead (cfg.mt_hybrid_preview) and this model's line replaces
    the draft once, when the sentence is finished.
  - Not on the recognition thread. The pipeline calls this from a worker, so
    a slow reply costs a late caption and never audio.

What is left is one call per finished utterance. Utterances are seconds apart,
so the duty cycle is low, and the added lag lands on a line whose draft is
already on screen.

Ollama rather than an in-process runtime because it costs no new dependency -
requests is already pinned - and because it holds the weights across engine
switches, which is what makes changing models in Settings a picker change
instead of a multi-gigabyte load inside this process. It is a local server on
this machine; nothing here talks to any other host.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections import deque
from urllib.parse import urlparse

import requests

from . import glossary
from .config import Config
from .translate import TranslationUnavailable

log = logging.getLogger(__name__)

# Belt and braces for a build that inlines reasoning in the content instead of
# splitting it into the "thinking" field - the parser _post relies on is the
# first line of defence, this catches a tagged block that slips past it. DOTALL
# because the block spans lines; the unterminated form catches a reply that
# llm_max_tokens cut off mid-thought.
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL)

# Each rule is here because its absence was a visible failure:
#
#   "the way a native speaker would say it" - without it small models
#   transliterate structure: 頭が痛い becomes "My head is painful".
#
#   The subject rule - Japanese omits the subject in most conversational
#   sentences. Left alone, a model writes "it" for all of them; told to use
#   the earlier lines, it usually gets the right person.
#
#   "never invent an ending" - the line is live speech and is sometimes cut
#   mid-sentence. A translator's instinct is to hand back a complete
#   sentence, and completing someone's sentence for them is a hallucination
#   the viewer has no way to detect.
#
#   "recognition mistakes" - the input is ASR output: homophones swapped, no
#   punctuation. Saying so stops the model translating a mis-heard word with
#   great fidelity when the intended one is obvious from context.
_SYSTEM = """You translate live {src} speech into natural English subtitles.

Each message is one line of {src}, produced by speech recognition from a video, stream, game or conversation. Reply with its English translation and nothing else.

Rules:
- Translate the meaning, the way a native English speaker would say it. Never translate word for word.
- {src} drops subjects and objects. Recover them from the earlier lines. If it is still unclear, pick the most natural reading.
- Match the speaker's tone: casual stays casual, polite stays polite, rough stays rough. Render slang, internet, anime and gaming terms with their natural English equivalents.
- {honorifics}
- Render names consistently with how you rendered them in earlier lines.
- The line may be cut off mid-sentence, lack punctuation, or contain recognition mistakes. Translate what is there and stop. Never invent an ending or add anything that was not said.
- Output ONLY the English line. No quotes, notes, romaji, alternatives or explanations. If the line carries no translatable content, output nothing at all."""

_HONORIFICS_KEEP = (
    "Keep honorifics attached to names as they are (-san, -chan, -kun, -sama, "
    "senpai, sensei)."
)
_HONORIFICS_DROP = (
    "Drop honorifics, or replace them with a natural English form of address."
)

# Appended when the Names box has something in it.
#
# This is where a name is actually decided. The transcriber usually hears the
# kanji correctly - it is the translator that has to pick a reading for them,
# and 天音 has several plausible ones, so it guesses and guesses consistently
# wrong. Telling it the reading once fixes every later line, and costs a
# handful of tokens on a prompt that is cached across calls anyway.
_GLOSSARY = """

Names and terms used in this audio. Use these exact spellings, and these
readings for any kanji given one:
{terms}"""

# Follows a line that contains stream or net slang - see glossary.py. In
# brackets and after the line, because this is a note to the translator and
# the line is what is to be translated.
_NOTE = "\n\n[Translator's note, not part of the line. {notes}]"

# The pipeline speaks ISO codes; the prompt reads better with a name.
_LANG_NAME = {"ja": "Japanese", "zh": "Chinese", "ko": "Korean", "en": "English"}

# Quote pairs an instruction model wraps a translation in despite being told not
# to. Straight, Japanese, and single.
_QUOTES = (('"', '"'), ("「", "」"), ("'", "'"), ("“", "”"))

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in _LOCAL_HOSTS


def find_ollama() -> str | None:
    """Path of the ollama executable, if it is installed."""
    found = shutil.which("ollama")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = os.path.join(local, "Programs", "Ollama", "ollama.exe")
        if os.path.exists(candidate):
            return candidate
    return None


def server_reachable(endpoint: str, timeout: float = 1.5) -> bool:
    try:
        requests.get(f"{endpoint.rstrip('/')}/api/tags", timeout=timeout).raise_for_status()
        return True
    except requests.RequestException:
        return False


def start_server(endpoint: str, wait_sec: float = 12.0) -> bool:
    """Launch `ollama serve` if it is installed but not running.

    Ollama normally runs from the tray, but it does not start with Windows
    unless told to, and "translation unavailable" is a poor answer to a
    problem this app can fix by itself. Only ever done for a local endpoint:
    a remote one being down is not something starting a process here helps.
    """
    if not _is_local(endpoint):
        return False
    exe = find_ollama()
    if exe is None:
        return False
    try:
        subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0),
        )
    except OSError as exc:
        log.warning("could not start ollama: %s", exc)
        return False
    deadline = time.monotonic() + wait_sec
    while time.monotonic() < deadline:
        if server_reachable(endpoint, timeout=1.0):
            log.info("started ollama serve")
            return True
        time.sleep(0.4)
    return False


def installed_models(endpoint: str, timeout: float = 3.0) -> list[str]:
    """Tags the local Ollama has pulled. Empty if it cannot be asked."""
    try:
        resp = requests.get(f"{endpoint.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return sorted(m.get("name", "") for m in resp.json().get("models", []))
    except (requests.RequestException, ValueError):
        return []


def has_model(endpoint: str, model: str) -> bool:
    names = set(installed_models(endpoint))
    # Ollama reports "qwen3:8b"; a name given without a tag means ":latest".
    return model in names or f"{model}:latest" in names


def pull_model(endpoint: str, model: str, on_progress=None, should_stop=None) -> None:
    """Download a model through Ollama, reporting progress as (fraction, text).

    Raises TranslationUnavailable on any failure. The download is done by the
    local Ollama server, from its registry or from Hugging Face for hf.co/
    tags - the one moment this app causes network traffic, and only when the
    user has asked for a model that is not installed.
    """
    url = f"{endpoint.rstrip('/')}/api/pull"
    try:
        with requests.post(
            url, json={"model": model, "stream": True}, stream=True, timeout=(5, 600)
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if should_stop is not None and should_stop():
                    return
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if chunk.get("error"):
                    raise TranslationUnavailable(
                        f"Ollama could not download {model}: {chunk['error']}"
                    )
                if on_progress is not None:
                    total, done = chunk.get("total"), chunk.get("completed")
                    fraction = done / total if total and done is not None else None
                    on_progress(fraction, chunk.get("status", ""))
    except requests.RequestException as exc:
        raise TranslationUnavailable(
            f"Could not download {model} through Ollama: {exc}"
        ) from exc


class LLMTranslator:
    """Same surface as translate.Translator: construct, then translate(str).

    Constructing it verifies the server is up and the model is pulled, then
    loads the weights, so a missing Ollama or an unpulled tag surfaces as a
    clear message at Start rather than as silently empty captions later.
    """

    name = "llm"
    uses_context = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_error: str | None = None
        self._url = cfg.llm_endpoint.rstrip("/")
        self._system = _SYSTEM.format(
            src=_LANG_NAME.get(cfg.mt_source_lang, cfg.mt_source_lang),
            honorifics=_HONORIFICS_KEEP if cfg.llm_keep_honorifics else _HONORIFICS_DROP,
        )
        # The Names box. It reaches Whisper too, except on the models a
        # prompt degrades - see Transcriber._asr_prompt - so on those this is
        # the only thing reading it.
        terms = cfg.initial_prompt.strip()
        if terms:
            self._system += _GLOSSARY.format(terms=terms)
        # (source, english) for the last llm_context_utterances calls, shown to
        # the model as prior turns. Only successful pairs go in: feeding back a
        # blank translation teaches it that blank is an acceptable answer.
        self._history: deque[tuple[str, str]] = deque(
            maxlen=max(1, cfg.llm_context_utterances)
        )
        self._glossary = glossary.default() if cfg.slang_glossary else None
        self._session = requests.Session()

        t0 = time.perf_counter()
        self._check_server()
        self._warmup()
        log.info(
            "loaded MT %s via %s in %.1fs",
            cfg.llm_model,
            self._url,
            time.perf_counter() - t0,
        )

    # -- startup checks ----------------------------------------------------

    def _check_server(self) -> None:
        """Confirm the server answers and the model is present, with the fix.

        Both messages say what to do, because this is the one failure the NMT
        path never had: Sugoi v4 downloads itself on first use, while this
        depends on a separate program being installed and running.
        """
        if not server_reachable(self._url) and not start_server(self._url):
            if _is_local(self._url) and find_ollama() is None:
                raise TranslationUnavailable(
                    "This translation model runs through Ollama, which is not "
                    "installed. Install it from ollama.com, or switch to the "
                    "Fast profile, which needs nothing else."
                )
            raise TranslationUnavailable(
                f"Ollama is not answering at {self._url}. Start Ollama and "
                "press Start again."
            )
        if not has_model(self._url, self.cfg.llm_model):
            raise TranslationUnavailable(
                f"The model {self.cfg.llm_model} has not been downloaded yet. "
                "Use Download in Settings, or run: "
                f"ollama pull {self.cfg.llm_model}"
            )

    def _warmup(self) -> None:
        """Force the weights resident, so the first real utterance isn't late.

        A cold Ollama loads on the first request, which for a 14B is several
        seconds. That would land on the first caption of the session - the one
        most likely to be watched - so it is paid here instead, behind the
        "loading translation model" status the pipeline is already showing.
        The timeout is generous for the same reason: this is a model load.
        """
        try:
            self._post([{"role": "user", "content": "こんにちは"}], timeout=180.0)
        except Exception as exc:  # noqa: BLE001 - a slow warmup must not block the run
            log.warning("LLM warmup failed, continuing: %s", exc)

    def warmup(self) -> None:
        """Already done during construction; present so both backends match."""

    def reset(self) -> None:
        """Forget the conversation so far."""
        self._history.clear()

    def amend(self, source: str, new_source: str, new_english: str) -> None:
        """Replace a remembered line with its corrected version.

        The second look calls this when it fixes a line, so that the next
        lines are translated with the right words as their context rather
        than the misheard ones.
        """
        for index, (src, _eng) in enumerate(self._history):
            if src == source:
                self._history[index] = (new_source, new_english)
                return

    def close(self) -> None:
        self._session.close()

    def unload(self) -> None:
        """Ask Ollama to release the model's memory now."""
        try:
            self._session.post(
                f"{self._url}/api/generate",
                json={"model": self.cfg.llm_model, "keep_alive": 0},
                timeout=10.0,
            )
        except requests.RequestException as exc:
            log.debug("could not unload %s: %s", self.cfg.llm_model, exc)

    # -- request -----------------------------------------------------------

    def _payload(self, messages: list[dict], stream: bool) -> dict:
        payload = {
            "model": self.cfg.llm_model,
            "messages": [{"role": "system", "content": self._system}] + messages,
            "stream": stream,
            "keep_alive": self.cfg.llm_keep_alive,
            "options": {
                "temperature": self.cfg.llm_temperature,
                "num_predict": self.cfg.llm_max_tokens,
                "num_ctx": self.cfg.llm_num_ctx,
                # Fixed, so the same line translated twice reads the same
                # twice; with a moving seed a re-run of a benchmark is not a
                # re-run.
                "seed": 7,
            },
        }
        # think=false is deliberately NOT sent, and sending it was the bug this
        # guards against. On a reasoning model it does not mean "do not think":
        # ollama's qwen3 template prefills <think> on every assistant turn
        # unconditionally, with no enable_thinking branch, so the model reasons
        # either way. All the flag decides is whether ollama parses that block
        # out - and false turns the parser off, which sends the deliberation
        # down "content" with its opening tag already eaten by the template.
        # _clean cannot strip a block whose tag never arrives, so it rendered
        # verbatim as captions. Omitted, the parser stays on and reasoning lands
        # in "thinking", which is read below and dropped.
        if self.cfg.llm_think:
            payload["think"] = True
        return payload

    def _post(self, messages: list[dict], timeout: float) -> dict:
        """One /api/chat round trip. Returns the raw assistant message object.

        The whole message rather than its content because a reasoning model
        splits its reply in two, and translate() has to be able to tell an
        empty answer apart from an answer that was crowded out by reasoning.
        """
        resp = self._session.post(
            f"{self._url}/api/chat", json=self._payload(messages, False), timeout=timeout
        )
        resp.raise_for_status()
        return resp.json().get("message") or {}

    def _post_stream(self, messages: list[dict], timeout: float, on_partial) -> dict:
        """_post, but handing each delta to on_partial as it arrives.

        Returns the same {"content", "thinking"} shape _post does, assembled
        from the chunks, so translate() cannot tell the two apart.

        The loop deliberately does not break on the done chunk. Stopping early
        leaves the body unread, and requests then drops the connection instead
        of returning it to the pool - measured, that made every following call
        wait ~1.7 s before its first token while the server finished with the
        socket that was walked away from. Ollama ends the stream after done, so
        letting the iterator run out costs nothing and keeps the connection
        reusable.
        """
        content: list[str] = []
        thinking: list[str] = []
        with self._session.post(
            f"{self._url}/api/chat",
            json=self._payload(messages, True),
            timeout=timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    # ollama emits one complete object per line. A partial one
                    # means the stream is truncated, not that this line is bad,
                    # so skipping it keeps whatever already arrived.
                    continue
                message = chunk.get("message") or {}
                if message.get("thinking"):
                    thinking.append(message["thinking"])
                piece = message.get("content") or ""
                if piece:
                    content.append(piece)
                    partial = self._clean_partial("".join(content))
                    if partial:
                        on_partial(partial)
        return {"content": "".join(content), "thinking": "".join(thinking)}

    def _messages(self, text: str) -> list[dict]:
        """The utterance, preceded by recent ones as completed turns."""
        msgs: list[dict] = []
        if self.cfg.llm_context_utterances > 0:
            for src, eng in self._history:
                msgs.append({"role": "user", "content": src})
                msgs.append({"role": "assistant", "content": eng})
        msgs.append({"role": "user", "content": self._with_notes(text)})
        return msgs

    def _with_notes(self, text: str) -> str:
        """The line, followed by what any slang in it means.

        Only the line itself goes into the history, so a note is shown once,
        with the line it is about, and does not ride along as context.
        """
        if self._glossary is None:
            return text
        notes = self._glossary.notes(text)
        if not notes:
            return text
        return text + _NOTE.format(notes="; ".join(notes))

    @staticmethod
    def _clean_partial(reply: str) -> str:
        """What is safe to show of a reply that has not finished arriving.

        _clean cannot be used mid-stream. It unwraps a leading quote whose
        closing partner may not have been generated yet, which would make the
        preview flicker in and out of its own quotes, and it strips a <think>
        block that may still be open.

        So: nothing at all while a think block is unclosed - the deliberation
        is not a translation and must never reach the screen - and no quote
        handling until the final _clean, which sees the whole reply.
        """
        if "<think>" in reply and "</think>" not in reply:
            return ""
        return _THINK.sub("", reply).lstrip()

    @staticmethod
    def _clean(reply: str) -> str:
        """Strip reasoning and the wrappers instruction models like to add."""
        reply = _THINK.sub("", reply)
        reply = _THINK_OPEN.sub("", reply)
        reply = reply.strip()
        # A second line is an afterthought - a note, an alternative, romaji -
        # and never part of a one-line subtitle.
        if "\n" in reply:
            reply = reply.split("\n", 1)[0].strip()
        # Only unwrap when the whole line is one quoted span, so a translation
        # that legitimately contains a quote is left alone.
        for open_q, close_q in _QUOTES:
            if (
                len(reply) > 1
                and reply.startswith(open_q)
                and reply.endswith(close_q)
                and close_q not in reply[1:-1]
            ):
                reply = reply[1:-1].strip()
                break
        return reply

    def translate(self, text: str, on_partial=None, beam_size: int | None = None) -> str:
        """Translate one utterance. Returns "" for empty or failed input.

        on_partial, when given, is called with the English so far as it
        streams. It is only ever a preview: the return value is the finished
        line, cleaned, and is what the transcript keeps.

        beam_size is accepted for parity with the NMT backend and ignored.

        A failure sets last_error and returns "". The pipeline reads that to
        tell "the model had nothing to say" from "the model could not be
        reached", which need opposite responses.
        """
        text = text.strip()
        if not text:
            return ""
        try:
            if on_partial is not None:
                message = self._post_stream(
                    self._messages(text), self.cfg.llm_timeout_sec, on_partial
                )
            else:
                message = self._post(
                    self._messages(text), timeout=self.cfg.llm_timeout_sec
                )
            english = self._clean(message.get("content") or "")
            self.last_error = None
            # Nothing but reasoning came back. On a model whose template forces
            # thinking, llm_max_tokens is spent deliberating before the answer
            # starts, so every caption is blank - a silent failure worth naming
            # once per utterance, with both fixes in the message.
            if not english and (message.get("thinking") or "").strip():
                self.last_error = (
                    f"{self.cfg.llm_model} spent its whole reply reasoning. "
                    "Pick a non-reasoning model such as qwen3:4b-instruct."
                )
                log.warning("%s", self.last_error)
        except requests.Timeout:
            # Off-thread, so this cost a caption and not the audio behind it.
            self.last_error = (
                f"The translation model took longer than "
                f"{self.cfg.llm_timeout_sec:.0f}s to answer."
            )
            log.warning("%s", self.last_error)
            return ""
        except requests.RequestException as exc:
            self.last_error = f"Lost contact with Ollama ({type(exc).__name__})."
            log.warning("LLM request failed: %s", exc)
            return ""
        except Exception as exc:  # noqa: BLE001 - a bad utterance must not kill the pipeline
            log.exception("translation failed for %r", text[:80])
            self.last_error = f"{type(exc).__name__}: {exc}"
            return ""

        if english:
            self._history.append((text, english))
        return english
