"""LLM translation stage: source-language text -> English, via a local Ollama.

Why this exists alongside translate.py. Sugoi is a 300M NMT model that sees one
sentence and nothing else, and its two standing failures both come from that:
it renames the same character on every line, and it cannot recover a subject the
speaker dropped - which Japanese speakers do constantly, so a good share of the
utterances on a stream arrive without one. Neither is fixable by tuning a model
that has no memory to tune.

An instruction model has the memory. It is shown the last few utterances as
prior turns, so it has already seen how it rendered a name and who is being
talked about, and it holds both. That is what is being bought here, and the
price is roughly 10x per call - Sugoi runs 46 ms, this runs a few hundred.

Where that price is affordable is the whole design:

  - Not on the live preview as the ct2 route does it - re-translating the tail
    every pass, ~0.35 s apart, on a budget of tens of milliseconds. Here
    mt_live_enabled instead streams the one utterance call as it arrives, so
    the preview costs no extra requests.
  - Not on the ASR thread. Half a second inline stalls the capture loop and the
    backlog is eventually discarded, so a slow translator would cost audio
    rather than latency. The LLM engines set mt_async and the pipeline runs them
    on a worker.

What is left is one call per committed utterance, arriving a few hundred
milliseconds after the source text does. Utterances are seconds apart, so the
duty cycle is low, and the added lag lands on a caption that was already waiting
for an utterance boundary.

Ollama rather than an in-process runtime because it costs no new dependency -
requests is already pinned - and because it holds the weights across engine
switches, which is what makes swapping 4B for 8B in the GUI a picker change
instead of a model load.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import deque

import requests

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

_SYSTEM = """You are a live subtitle translator. Translate each {src} line into natural English.

Rules:
- Output ONLY the English translation. No notes, no romaji, no quotes, no explanation.
- The line comes from live speech and may be cut off mid-sentence. Translate exactly what is there and stop. Never invent an ending.
- Keep the speaker's register: casual speech stays casual.
- Render names consistently with how you rendered them in earlier lines.
- If the line carries no translatable content, output nothing at all."""

# Appended when the Names box has something in it.
#
# This is where a name is actually decided. The transcriber usually hears the
# kanji correctly - it is the translator that has to pick a reading for them,
# and 天音 has several plausible ones, so it guesses and guesses consistently
# wrong. Telling it the reading once fixes every later line, and costs a
# handful of tokens on a prompt that is cached across calls anyway.
_GLOSSARY = """

Names and terms used in this stream. Use these exact spellings, and these
readings for any kanji given one:
{terms}"""

# The pipeline speaks ISO codes; the prompt reads better with a name. Only the
# four the GUI offers are here - see LANGUAGES in window.py.
_LANG_NAME = {"ja": "Japanese", "zh": "Chinese", "ko": "Korean", "en": "English"}

# Quote pairs an instruction model wraps a translation in despite being told not
# to. Straight, Japanese, and single.
_QUOTES = (('"', '"'), ("「", "」"), ("'", "'"))


class LLMTranslator:
    """Same surface as translate.Translator: construct, then translate(str).

    Constructing it verifies the server is up and the model is pulled, then
    loads the weights, so a missing Ollama or an unpulled tag surfaces as a
    status line at Apply rather than as silently empty captions later.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._url = cfg.llm_endpoint.rstrip("/")
        self._system = _SYSTEM.format(
            src=_LANG_NAME.get(cfg.mt_source_lang, cfg.mt_source_lang)
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

        Both messages name the command that fixes them, because this is the one
        failure the ct2 path never had: Sugoi downloads itself on first use,
        while this depends on a service the user has to have started.
        """
        try:
            resp = self._session.get(f"{self._url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            tags = resp.json().get("models", [])
        except requests.RequestException as exc:
            raise TranslationUnavailable(
                f"no Ollama at {self._url} ({exc}) - install it, then "
                f"'ollama serve'"
            ) from exc
        except ValueError as exc:
            raise TranslationUnavailable(
                f"{self._url} answered but is not Ollama"
            ) from exc

        # Ollama reports "qwen3:8b"; a name given without a tag means ":latest".
        names = {m.get("name", "") for m in tags}
        want = self.cfg.llm_model
        if want not in names and f"{want}:latest" not in names:
            raise TranslationUnavailable(
                f"model {want} not pulled - run 'ollama pull {want}'"
            )

    def _warmup(self) -> None:
        """Force the weights resident, so the first real utterance isn't late.

        A cold Ollama loads on the first request, which for an 8B is seconds.
        That would land on the first caption of the session - the one most
        likely to be watched - so it is paid here instead, behind the "loading
        translation model..." status the pipeline is already showing. The
        timeout is generous for the same reason: this is a model load.
        """
        try:
            self._post([{"role": "user", "content": "こんにちは"}], timeout=180.0)
        except Exception as exc:  # noqa: BLE001 - a slow warmup must not block the run
            log.warning("LLM warmup failed, continuing: %s", exc)

    # -- request -----------------------------------------------------------

    def _post(self, messages: list[dict], timeout: float) -> dict:
        """One /api/chat round trip. Returns the raw assistant message object.

        The whole message rather than its content because a reasoning model
        splits its reply in two, and translate() has to be able to tell an
        empty answer apart from an answer that was crowded out by reasoning.
        """
        payload = {
            "model": self.cfg.llm_model,
            "messages": [{"role": "system", "content": self._system}] + messages,
            "stream": False,
            "keep_alive": self.cfg.llm_keep_alive,
            "options": {
                "temperature": self.cfg.llm_temperature,
                "num_predict": self.cfg.llm_max_tokens,
                "num_ctx": self.cfg.llm_num_ctx,
            },
        }
        # think=false is deliberately NOT sent, and sending it was the bug this
        # guards against. On a reasoning model it does not mean "do not think":
        # ollama 0.33's qwen3 template prefills <think> on every assistant turn
        # unconditionally, with no enable_thinking branch, so the model reasons
        # either way. All the flag decides is whether ollama parses that block
        # out - and false turns the parser off, which sends the deliberation
        # down "content" with its opening tag already eaten by the template.
        # _clean cannot strip a block whose tag never arrives, so it rendered
        # verbatim as captions. Omitted, the parser stays on and reasoning lands
        # in "thinking", which is read below and dropped.
        if self.cfg.llm_think:
            payload["think"] = True
        resp = self._session.post(
            f"{self._url}/api/chat", json=payload, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json().get("message") or {}

    def _post_stream(self, messages: list[dict], timeout: float, on_partial) -> dict:
        """_post, but handing each delta to on_partial as it arrives.

        Returns the same {"content", "thinking"} shape _post does, assembled
        from the chunks, so translate() cannot tell the two apart.

        This costs no extra requests: it is the one call that was already
        being made, read as it streams instead of after it finishes.

        The loop deliberately does not break on the done chunk. Stopping early
        leaves the body unread, and requests then drops the connection instead
        of returning it to the pool - measured, that made every following call
        wait ~1.7 s before its first token while the server finished with the
        socket that was walked away from. Ollama ends the stream after done, so
        letting the iterator run out costs nothing and keeps the connection
        reusable.
        """
        payload = {
            "model": self.cfg.llm_model,
            "messages": [{"role": "system", "content": self._system}] + messages,
            "stream": True,
            "keep_alive": self.cfg.llm_keep_alive,
            "options": {
                "temperature": self.cfg.llm_temperature,
                "num_predict": self.cfg.llm_max_tokens,
                "num_ctx": self.cfg.llm_num_ctx,
            },
        }
        # See the note in _post: think is sent only when it is wanted.
        if self.cfg.llm_think:
            payload["think"] = True

        content: list[str] = []
        thinking: list[str] = []
        with self._session.post(
            f"{self._url}/api/chat", json=payload, timeout=timeout, stream=True
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
        msgs.append({"role": "user", "content": text})
        return msgs

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

    def translate(self, text: str, on_partial=None) -> str:
        """Translate one utterance. Returns "" for empty or failed input.

        on_partial, when given, is called with the English so far as it
        streams. It is only ever a preview: the return value is the finished
        line, cleaned, and is what the transcript keeps.
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
            # Nothing but reasoning came back. On a model whose template forces
            # thinking, llm_max_tokens is spent deliberating before the answer
            # starts, so every caption is blank - a silent failure worth naming
            # once per utterance, with both fixes in the message.
            if not english and (message.get("thinking") or "").strip():
                log.warning(
                    "%s spent the whole %d-token budget reasoning - raise "
                    "llm_max_tokens or pick a non-reasoning model such as "
                    "qwen3:4b-instruct",
                    self.cfg.llm_model,
                    self.cfg.llm_max_tokens,
                )
        except requests.Timeout:
            # Off-thread, so this cost a caption and not the audio behind it.
            log.warning("LLM timed out after %.0fs", self.cfg.llm_timeout_sec)
            return ""
        except Exception:  # noqa: BLE001 - a bad utterance must not kill the pipeline
            log.exception("translation failed for %r", text[:80])
            return ""

        if english:
            self._history.append((text, english))
        return english
