"""Stream and net slang the translation models do not reliably know.

A translator that has never met てぇてぇ renders it as whatever it sounds
like, and one that knows 枠 only as "frame" turns "next stream" into "next
frame". Neither is a failure of context, so no amount of earlier lines fixes
it: the word simply is not in the model. It is in this file instead.

The list is looked up, not pasted. Showing a model a hundred definitions on
every line costs tokens and invites it to use them - a translator told what
草 means starts finding things funny. So each line is searched for the terms
it actually contains, and only those few are put in front of the model, as a
note attached to that line.

What goes in the list:

  - Words whose stream meaning differs from the dictionary one (枠, 凸, 箱推し,
    卒業, ママ), or that a dictionary does not have at all (てぇてぇ, スパチャ).
  - Game and gacha vocabulary that small models translate literally (天井,
    爆死, 漁夫, 芋る).
  - Not ordinary casual Japanese. やばい and マジ are in every model already;
    a note about them would only be noise.

Each entry lists the forms the word is written in and, for a short word that
is usually something else, the compounds that are not the slang (草 in 雑草
is grass).

An entry can also list what the recogniser writes when it has never met the
word: 凸待ち comes out as 突待ち, 同接 as 同説, てぇてぇ as ていてい - a real
sound spelt with whatever characters the model does know. Those are not
alternative spellings, they are mistakes with a known answer, so they are
put right in the text itself (respell) before anyone translates it. That is
the one repair to the Japanese that needs no model and no context: the wrong
spelling is not a word, so there is nothing else it could have been.

Matching folds katakana to hiragana and small kana to full size, so スパチャ
and すぱちゃ, てぇてぇ and てえてえ, are one form. It is deterministic: the
same line always gets the same notes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# How many notes one line may carry. Past a handful the note is longer than
# the line, and a line with six slang terms in it is rare enough that the cap
# only ever bites on a false match.
MAX_NOTES = 5


@dataclass(frozen=True)
class Term:
    # Written forms, the first being the one shown to the model.
    forms: tuple[str, ...]
    # What it means, phrased so it can follow "X:" in a prompt.
    meaning: str
    # Compounds containing a form that are not this word.
    unless: tuple[str, ...] = ()
    # What a recogniser writes instead of forms[0] when it does not know it.
    heard: tuple[str, ...] = ()
    # Match in the script given, not in any: ママ is not まま.
    exact: bool = False


def _t(forms: str, meaning: str, unless: str = "", exact: bool = False) -> Term:
    """An entry. In `forms`, a leading ~ marks a mishearing rather than a spelling."""
    parts = [f for f in forms.split("|") if f]
    return Term(
        tuple(f for f in parts if not f.startswith("~")),
        meaning,
        tuple(u for u in unless.split("|") if u),
        tuple(f[1:] for f in parts if f.startswith("~")),
        exact,
    )


TERMS: tuple[Term, ...] = (
    # -- reactions and chat ------------------------------------------------
    _t("草|草生える|草はえる|草生えた|草不可避", '"lol" - a laughing reaction, not grass',
       "雑草|草原|草花|草木|薬草|草食|草刈|草むら|草野球|草履|海草|牧草|煙草|草案|起草|草むしり|草取り"),
    _t("大草原", '"LMAO" - laughing hard'),
    _t("ワロタ|ワロス|わろた", '"lol"'),
    _t("てぇてぇ|てえてえ|てぇてー|~ていてい|~テーテー", '"precious" / "wholesome" - said of a heartwarming moment '
       "between two people (from 尊い)"),
    _t("尊い", 'in fan talk: "precious", too lovely for words', "尊い命|尊い犠牲"),
    _t("それな", '"exactly" / "so true"', "それなら|それなの|それなり|それなん|それなく|それなし"),
    _t("わかりみ", '"so relatable"'),
    _t("エモい", '"emotional", "hits you in the feels"'),
    _t("ぴえん", 'cutesy crying: "boo-hoo"'),
    _t("ワンチャン", '"there\'s a chance" / "maybe"'),
    _t("しか勝たん", '"X しか勝たん" = "nothing beats X", "X is the best"'),
    _t("ないなった", '"it\'s gone" (cute for なくなった)'),
    _t("うぽつ", '"thanks for the upload"'),
    _t("わこつ", '"thanks for starting the stream"'),
    _t("初見", '"first-time viewer" (of a game: "playing it blind")'),
    _t("古参", '"long-time fan"'),
    _t("ロム専|ROM専", '"lurker"'),
    _t("荒らし", '"troll"'),
    _t("指示厨|~支持厨", '"backseater" - a viewer who tells the streamer how to play'),
    _t("指示コメ|~支持コメ", '"backseating comments"'),
    _t("ネタバレ", '"spoilers"'),
    _t("鳩行為|伝書鳩|鳩コメ", "mentioning what another streamer is doing in this chat "
       '(carrying messages between streams): "hatoing"'),
    _t("杞憂民", '"viewers who worry too much"'),
    _t("厄介オタク|厄介オタ|厄介リスナー", '"obnoxious fan"'),
    _t("限界オタク|限界オタ", '"a fan who is completely overwhelmed" (down bad)'),
    _t("ガチ恋", '"gachikoi" - a fan genuinely in love with the streamer'),
    _t("解釈違い", '"that\'s not how I see the character"'),
    _t("解釈一致", '"exactly how I imagined the character"'),
    _t("お気持ち表明|お気持ち長文", '"posting a long complaint about one\'s feelings"'),
    _t("炎上", '"backlash" / "getting flamed online"', "炎上する建物"),
    _t("垢バン|垢BAN|アカバン", '"account ban"'),
    _t("陰キャ", '"gloomy introvert"'),
    _t("陽キャ", '"outgoing, popular type"'),
    _t("リア充", '"normie" - someone with a fulfilling real-life social life'),
    # -- streaming ---------------------------------------------------------
    _t("スパチャ|スーパーチャット|スパチャット", '"Super Chat" - a paid, highlighted chat message'),
    _t("赤スパ|赤すぱ", '"red Super Chat" - the most expensive tier'),
    _t("メン限|メンバー限定", '"members-only" (stream)'),
    _t("メンバーシップ|メンシ", '"channel membership"'),
    _t("枠", 'a stream, or its time slot: 次の枠 "the next stream", 枠を取る "to schedule a stream"',
       "枠組|窓枠|枠内|枠外|枠線|別枠|大枠|外枠|内枠|木枠|枠にとらわれ|枠を超え"),
    _t("歌枠", '"karaoke stream" / "singing stream"'),
    _t("雑談枠|雑談配信", '"chatting stream" / "free talk"'),
    _t("朝枠|朝活", '"morning stream"'),
    _t("ゲリラ配信|ゲリラ枠|ゲリラ", '"surprise stream" - unannounced'),
    _t("耐久配信|耐久枠|耐久", '"endurance stream" - going until a goal is reached',
       "耐久性|耐久力|耐久値|耐久度|耐久レース"),
    _t("凸待ち|とつ待ち|~突待ち", '"call-in stream" - the streamer waits for others to call in'),
    _t("逆凸|逆とつ|~逆突", '"reverse call-in" - the streamer calls others unannounced'),
    _t("凸る|凸った|凸して|凸します|凸",
       'to drop in on someone\'s stream or call unannounced: "call in"', "凸凹|凸レンズ|凸版"),
    _t("同接|~同説", '"concurrent viewers"'),
    _t("リスナー", '"viewers" / "chat"'),
    _t("切り抜き", '"clip" - a short highlight cut from a stream', "新聞の切り抜き|切り抜き帳"),
    _t("切り抜き師", '"clipper"'),
    _t("アーカイブ", '"the VOD" / "the archive" of a stream'),
    _t("待機所|待機枠", '"waiting room" before a stream starts'),
    _t("配信事故|放送事故", '"on-stream accident"'),
    _t("切り忘れ", '"forgot to end the stream"'),
    _t("寝落ち", '"fell asleep" in the middle of a stream or call'),
    _t("台パン", '"desk slam" - hitting the desk in frustration'),
    _t("案件", 'of a streamer: a "sponsored" stream or job', "案件数|この案件について"),
    _t("収益化", '"monetization"'),
    _t("銀盾", '"Silver Play Button" (100,000 subscribers)'),
    _t("金盾", '"Gold Play Button" (1,000,000 subscribers)'),
    _t("高評価", '"a like" on the video'),
    _t("チャンネル登録", '"subscribe to the channel"'),
    _t("登録者", '"subscribers"'),
    _t("記念配信|記念枠", '"celebration stream"'),
    _t("生誕祭", '"birthday celebration"'),
    _t("視聴者参加型|参加型", '"viewer participation" stream - viewers can join the game'),
    _t("スナイプ|ゴースティング", '"stream sniping"'),
    _t("実況", '"let\'s play" / "live commentary"', "実況見分"),
    # -- VTubers and fandom ------------------------------------------------
    _t("ブイチューバー|Vチューバー|VTuber|ブイチューバ", '"VTuber"'),
    _t("ホロライブ", '"hololive" (VTuber agency)'),
    _t("ホロメン", '"holomem" - a hololive member'),
    _t("にじさんじ", '"Nijisanji" (VTuber agency)'),
    _t("ぶいすぽ", '"VSPO!" (VTuber esports group)'),
    _t("ライバー", '"streamer" - what Nijisanji calls its talents; not the organ'),
    _t("個人勢", '"indie VTuber"'),
    _t("企業勢", '"corporate VTuber" - one who belongs to an agency'),
    _t("推し", '"oshi" - the one you support most, your favourite', "推し進め|推し量"),
    _t("箱推し|~箱押し", '"supporting the whole group" rather than one member'),
    _t("単推し|~単押し", '"supporting one member only"'),
    _t("推し変|~押し変", '"switching to a different oshi"'),
    _t("推し活", '"oshi activities" - supporting your favourite'),
    _t("同期", 'of VTubers: "genmate" - someone who debuted in the same generation',
       "同期する|同期して|同期ずれ|同期が取れ"),
    _t("期生", 'N期生 = "Nth generation" (debut group)'),
    _t("卒業", 'of a VTuber or idol: "graduation" - leaving the agency and retiring the '
       "character", "卒業式|卒業証書|卒業論文|卒論|卒業旅行"),
    _t("転生", 'of a VTuber: "re-debuting as a new character"', "異世界転生|輪廻転生"),
    _t("前世", 'of a VTuber: "their previous identity" before this character'),
    _t("中の人", '"the person behind the character"'),
    _t("ママ", "of a VTuber: often the illustrator who designed their character; otherwise "
       '"mom"', exact=True),
    _t("絵師", '"illustrator" / "artist"'),
    _t("立ち絵", '"character art"'),
    _t("新衣装", '"new outfit" for the character model'),
    _t("お披露目|おひろめ|~お疲労目", '"reveal" / "showing off for the first time"'),
    _t("オフコラボ", '"offline collab" - streaming together in person'),
    _t("コラボ", '"collab"'),
    _t("清楚", '"seiso" - a pure, wholesome image'),
    _t("センシティブ", 'of a stream or its content: "lewd" / "risqué"; otherwise "sensitive"'),
    _t("あざとい", '"calculatedly cute"'),
    _t("ぽんこつ|ポンコツ", '"clumsy airhead"'),
    _t("バブみ", '"motherly vibes"'),
    _t("ファンサ", '"fan service"'),
    _t("ファンアート", '"fan art"'),
    _t("ボイス販売|シチュボ|シチュエーションボイス", '"voice pack" / "roleplay audio" for sale'),
    _t("グッズ", '"merch"'),
    _t("茶番", '"a skit" / "a farce"'),
    _t("罰ゲーム", '"punishment game" - a forfeit for losing'),
    _t("大喜利", '"ogiri" - improvised joke answers to a prompt'),
    # -- games -------------------------------------------------------------
    _t("神回", '"a legendary episode" / "an amazing stream"'),
    _t("神ゲー", '"a masterpiece of a game"'),
    _t("クソゲー|クソゲ", '"a trash game" (kusoge)'),
    _t("ホラゲー|ホラゲ", '"horror game"'),
    _t("沼る|沼った|沼って|沼り", '"to get stuck failing over and over" (or: to get hooked)'),
    _t("詰んだ|詰みです|詰みました|詰んで", '"I\'m stuck with no way out" / "I\'m done for"'),
    _t("縛りプレイ|縛りプレー", '"challenge run" with self-imposed restrictions'),
    _t("初見プレイ|初見プレー", '"blind playthrough"'),
    _t("RTA|アールティーエー", '"speedrun"'),
    _t("ガチ勢", '"hardcore players"'),
    _t("エンジョイ勢", '"casual players"'),
    _t("ガチャ", '"gacha" - a random-draw pull'),
    _t("爆死", 'in gacha: "spent a lot and got nothing"'),
    _t("神引き", '"a god-tier pull" - amazing gacha luck'),
    _t("天井", 'in gacha: "pity" - the number of pulls that guarantees the item',
       "天井裏|天井が高|天井に頭|天井から"),
    _t("課金", '"spending real money" on a game'),
    _t("無課金", '"free-to-play" (spending nothing)'),
    _t("廃課金", '"whale" - someone who spends huge amounts'),
    _t("周回", '"farming" / "grinding" runs'),
    _t("乱数", '"RNG"'),
    _t("ナーフ", '"nerf"'),
    _t("ぶっ壊れ", '"broken" / "overpowered"'),
    _t("ワンパン", '"one-shot"'),
    _t("芋る|芋って|芋ってる|芋プ", '"camping" (hiding in one spot in a shooter)'),
    _t("漁夫|漁夫る|漁夫られ", '"third-partying" - attacking two teams already fighting'),
    _t("立ち回り", '"positioning" / "game sense"'),
    _t("煽り|煽られ|煽って", '"taunting" / "trash talk"', "煽り運転"),
    _t("チーター", '"cheater"'),
    _t("脳死", 'in games: "mindlessly", "on autopilot"', "脳死判定"),
    _t("雑魚", '"small fry" / "weakling"'),
    _t("ドンマイ", '"don\'t worry about it"'),
    _t("キャリー", '"carry" (win the game for the team)', "キャリーケース|キャリーバッグ"),
    _t("エイム", '"aim"'),
)


_SMALL = str.maketrans("ぁぃぅぇぉゃゅょゎっ", "あいうえおやゆよわつ")


def _plain(text: str) -> str:
    """Width and case folded, script left alone."""
    return unicodedata.normalize("NFKC", text).casefold()


def fold(text: str) -> str:
    """The form terms are compared in: width, case, script and kana size folded."""
    text = unicodedata.normalize("NFKC", text).casefold()
    out = []
    for ch in text:
        code = ord(ch)
        # Katakana to hiragana. ー is left alone: it is not a letter.
        if 0x30A1 <= code <= 0x30F6:
            ch = chr(code - 0x60)
        out.append(ch)
    return "".join(out).translate(_SMALL)


class Glossary:
    def __init__(self, terms: tuple[Term, ...] = TERMS):
        self._terms = terms
        self._folded = [
            (
                tuple(_plain(f) if term.exact else fold(f) for f in term.forms + term.heard),
                tuple(_plain(u) if term.exact else fold(u) for u in term.unless),
            )
            for term in terms
        ]
        # (mishearing, proper spelling), longest first so that a mishearing
        # inside a longer one is not rewritten on its own.
        self._respell = sorted(
            ((heard, term.forms[0]) for term in terms for heard in term.heard),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        # One pass over the text to see whether anything can match at all,
        # which on most lines nothing does.
        self._any = re.compile(
            "|".join(
                sorted(
                    # Folded even for the entries matched exactly: this only
                    # has to say that a match is possible.
                    {re.escape(fold(f)) for forms, _u in self._folded for f in forms},
                    key=len,
                    reverse=True,
                )
            )
        )

    def lookup(self, text: str, limit: int = MAX_NOTES) -> list[tuple[str, Term]]:
        """(the term as written in `text`, term) for each term found.

        In the order they occur. A longer term hides a shorter one found
        inside it - 箱推し is not also 推し - and a form that only occurs
        inside one of its `unless` compounds is not a match.
        """
        folded = fold(text)
        if not folded or self._any.search(folded) is None:
            return []
        plain = _plain(text)
        hits: list[tuple[int, int, int]] = []  # (start, -length, term index)
        for index, (forms, unless) in enumerate(self._folded):
            haystack = plain if self._terms[index].exact else folded
            best: tuple[int, int] | None = None
            for form in forms:
                start = haystack.find(form)
                while start >= 0:
                    end = start + len(form)
                    if not self._excluded(haystack, start, end, unless):
                        if best is None or (start, -len(form)) < best:
                            best = (start, -len(form))
                        break
                    start = haystack.find(form, start + 1)
            if best is not None:
                hits.append((best[0], best[1], index))
        hits.sort()
        found: list[tuple[str, Term]] = []
        covered: list[tuple[int, int]] = []
        # Folding keeps one character per character for Japanese text, so a
        # position in the folded string is a position in the original. If it
        # did not, the term's own spelling stands in.
        aligned = len(folded) == len(text)
        for start, neg_len, index in hits:
            end = start - neg_len
            if any(s <= start and end <= e for s, e in covered):
                continue
            covered.append((start, end))
            term = self._terms[index]
            found.append((text[start:end] if aligned else term.forms[0], term))
            if len(found) >= limit:
                break
        return found

    @staticmethod
    def _excluded(text: str, start: int, end: int, unless: tuple[str, ...]) -> bool:
        for compound in unless:
            at = text.find(compound, max(0, start - len(compound)))
            while 0 <= at <= start:
                if at + len(compound) >= end:
                    return True
                at = text.find(compound, at + 1)
        return False

    def respell(self, text: str) -> str:
        """`text` with each known mishearing of a term written properly.

        Exact matches only, unlike lookup: these are specific wrong spellings
        with one right answer each, not words to be recognised in any script.
        """
        for heard, proper in self._respell:
            if heard in text:
                text = text.replace(heard, proper)
        return text

    def notes(self, text: str, limit: int = MAX_NOTES) -> list[str]:
        """One "term: meaning" line per term found in `text`.

        When the text spells the term some other way - the recogniser's
        doing - the note says so, which is what lets a reader of the note
        put the right word back.
        """
        out = []
        for written, term in self.lookup(text, limit):
            proper = term.forms[0]
            if fold(written) == fold(proper):
                out.append(f"{proper}: {term.meaning}")
            else:
                out.append(f"{proper} (misheard here as {written}): {term.meaning}")
        return out


_default: Glossary | None = None


def default() -> Glossary:
    """The built-in list, compiled once."""
    global _default
    if _default is None:
        _default = Glossary()
    return _default
