"""Découpe incrémentale de la réponse du LLM.

Format attendu (parole d'abord, pour que le TTS démarre au plus tôt) :

    <say lang="fr">Presque ! On dit <en>I went</en>, pas "I goed".</say>
    <fix>[{"type": "conjugation", "original": "I goed", ...}]</fix>

En leçon, une balise <lesson>{"section": 2}</lesson> entre </say> et <fix> dit où en est le prof
(voir server/lessons.py).

Le texte de <say> est découpé en segments (proposition ou changement de langue)
envoyés au TTS au fil de l'eau ; le JSON de <fix> est lu à la fin.
"""

import json
import re
from dataclasses import dataclass, field

LANGS = ("en", "fr")
# Types par défaut (anglais) ; chaque matière fournit les siens (subjects/*.yaml)
FIX_TYPES = ("grammar", "conjugation", "vocabulary", "word_order", "preposition", "false_friend")
# Mot mal écrit par le STT (« le chaîne » pour le chêne). Le LLM relève ces mots quoi qu'on lui dise ;
# on lui donne donc un type pour les ranger, et on les écarte : ni affichés, ni comptés en mémoire.
TRANSCRIPTION = "transcription"
_LANG_ATTR = re.compile(r"""lang\s*=\s*["']?(en|fr)""", re.IGNORECASE)
_TAG = re.compile(r"</?\s*(say|fix|en|fr|lesson)\b[^>]*>", re.IGNORECASE)
_HARD_END = ".!?…"
_SOFT_END = ",;:—"
_CLOSE_QUOTES = "\"'»”)"
_MAX_TAG_LEN = 24


@dataclass
class Segment:
    text: str
    lang: str


@dataclass
class ParsedReply:
    lang: str = "en"
    say_text: str = ""
    segments: list[Segment] = field(default_factory=list)
    fixes: list[dict] = field(default_factory=list)
    fix_parse_error: bool = False
    lesson: dict | None = None  # balise <lesson> : où en est le prof dans sa leçon


class ReplyParser:
    def __init__(self, default_lang: str = "en", fix_types: tuple[str, ...] = FIX_TYPES,
                 first_min_chars: int = 8, min_chars: int = 60):
        self.state = "pre"
        self.fix_types = fix_types
        self.default_lang = default_lang
        self.first_min_chars = first_min_chars
        self.min_chars = min_chars
        self._pending = ""
        self._seg = ""
        self._lang_stack: list[str] = [default_lang]
        self._fix_buf = ""
        self._lesson_buf = ""
        self._display: list[str] = []
        self.reply = ParsedReply()

    @property
    def lang(self) -> str:
        return self._lang_stack[-1]

    def feed(self, delta: str) -> tuple[list[Segment], str]:
        """Ajoute un morceau de flux. Renvoie (segments prêts pour le TTS, texte à afficher)."""
        self._pending += delta
        segments: list[Segment] = []
        display: list[str] = []
        self._consume(segments, display, final=False)
        return segments, "".join(display)

    def close(self) -> tuple[list[Segment], str]:
        """Fin du flux : vide les tampons et lit les corrections."""
        segments: list[Segment] = []
        display: list[str] = []
        self._consume(segments, display, final=True)
        if self.state in ("pre", "say"):
            self._flush(segments)
        if self._fix_buf:
            self._parse_fixes()
        if self._lesson_buf and self.reply.lesson is None:
            self._parse_lesson()
        self.reply.lang = self._lang_stack[0]
        self.reply.say_text = "".join(self._display).strip()
        return segments, "".join(display)

    def _consume(self, segments: list[Segment], display: list[str], final: bool) -> None:
        while self._pending:
            # On cherche la balise fermante dans tout le tampon : elle arrive souvent coupée entre deux
            # morceaux du flux (« </les » puis « son> »).
            if self.state == "fix":
                self._fix_buf += self._pending
                end = self._fix_buf.lower().find("</fix>")
                if end < 0:
                    self._pending = ""
                    return
                self._pending = self._fix_buf[end + len("</fix>"):]
                self._fix_buf = self._fix_buf[:end]
                self.state = "done"
                continue
            if self.state == "lesson":
                self._lesson_buf += self._pending
                end = self._lesson_buf.lower().find("</lesson>")
                if end < 0:
                    self._pending = ""
                    return
                self._pending = self._lesson_buf[end + len("</lesson>"):]
                self._lesson_buf = self._lesson_buf[:end]
                self.state = "post"
                self._parse_lesson()  # tout de suite : un prof coupé par l'élève garde son avancement
                continue
            if self.state == "done":
                self._pending = ""
                return

            lt = self._pending.find("<")
            if lt < 0:
                text, self._pending = self._pending, ""
                self._on_text(text, segments, display)
                return
            if lt > 0:
                text, self._pending = self._pending[:lt], self._pending[lt:]
                self._on_text(text, segments, display)
                continue
            # le tampon commence par "<"
            gt = self._pending.find(">")
            if gt < 0:
                if not final and len(self._pending) < _MAX_TAG_LEN:
                    return  # balise peut-être coupée entre deux morceaux
                text, self._pending = self._pending[0], self._pending[1:]
                self._on_text(text, segments, display)
                continue
            tag = self._pending[: gt + 1]
            m = _TAG.fullmatch(tag)
            if not m:
                text, self._pending = self._pending[0], self._pending[1:]
                self._on_text(text, segments, display)
                continue
            self._pending = self._pending[gt + 1:]
            self._on_tag(m.group(1).lower(), tag.startswith("</"), tag, segments)

    def _on_text(self, text: str, segments: list[Segment], display: list[str]) -> None:
        if self.state == "pre":
            if not text.strip():
                return
            self.state = "say"  # le LLM a oublié <say> : on lit quand même
        if self.state != "say":
            return
        self._seg += text
        self._display.append(text)
        display.append(text)
        self._split(segments)

    def _on_tag(self, name: str, closing: bool, raw: str, segments: list[Segment]) -> None:
        if name == "say":
            if closing:
                if self.state == "say":
                    self._flush(segments)
                    self.state = "post"
            elif self.state == "pre":
                m = _LANG_ATTR.search(raw)
                if m:
                    self._lang_stack = [m.group(1).lower()]
                    self.reply.lang = self._lang_stack[0]
                self.state = "say"
        elif name == "fix":
            if not closing:
                if self.state in ("pre", "say"):
                    self._flush(segments)
                self.state = "fix"
        elif name == "lesson":
            if not closing:
                if self.state in ("pre", "say"):
                    self._flush(segments)
                self.state = "lesson"
        elif self.state in ("say", "pre"):
            self.state = "say"
            if closing:
                if len(self._lang_stack) > 1 and self._lang_stack[-1] == name:
                    if self._lang_stack[-2] != name:
                        self._flush(segments)
                    self._lang_stack.pop()
            elif name != self.lang:
                self._flush(segments)
                self._lang_stack.append(name)
            else:
                self._lang_stack.append(name)

    def _split(self, segments: list[Segment]) -> None:
        while True:
            cut = self._find_boundary()
            if cut is None:
                return
            head, self._seg = self._seg[:cut], self._seg[cut:]
            self._emit(head, segments)

    def _find_boundary(self) -> int | None:
        seg = self._seg
        min_soft = self.first_min_chars if not self.reply.segments else self.min_chars
        for i, ch in enumerate(seg[:-1]):
            nxt = seg[i + 1]
            if ch == "\n":
                return i + 1
            if not nxt.isspace():
                continue
            hard = ch in _HARD_END or (ch in _CLOSE_QUOTES and i > 0 and seg[i - 1] in _HARD_END)
            if hard and len(seg[: i + 1].strip()) >= 2:
                return i + 1
            if ch in _SOFT_END and i + 1 >= min_soft:
                return i + 1
        return None

    def _flush(self, segments: list[Segment]) -> None:
        head, self._seg = self._seg, ""
        self._emit(head, segments)

    def _emit(self, text: str, segments: list[Segment]) -> None:
        # la ponctuation laissée en tête par une balise fermante (". Now…", ", pas…") ne se prononce pas
        text = " ".join(text.split()).lstrip(".,;:!?— ")
        if not any(c.isalnum() for c in text):
            return
        # Une balise <en>/<fr> fait foi ; hors balise, le LLM oublie parfois de marquer une phrase entière
        lang = self.lang if len(self._lang_stack) > 1 else guess_lang(text, self.lang)
        seg = Segment(text, lang)
        self.reply.segments.append(seg)
        segments.append(seg)

    def _parse_fixes(self) -> None:
        raw = self._fix_buf
        start, end = raw.find("["), raw.rfind("]")
        if start < 0:
            self.reply.fix_parse_error = bool(raw.strip())
            return
        if end <= start:  # crochet fermant oublié
            raw, end = raw.rstrip() + "]", len(raw.rstrip())
        try:
            items = json.loads(raw[start: end + 1])
        except json.JSONDecodeError:
            items = _repair_fixes(raw[start + 1: end])
            if not items:
                self.reply.fix_parse_error = True
                return
        fixes = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict) or not it.get("original") or not it.get("corrected"):
                continue
            kind = str(it.get("type", "grammar")).lower()
            if kind == TRANSCRIPTION:
                continue
            fixes.append({
                "type": kind if kind in self.fix_types else self.fix_types[0],
                "original": str(it["original"]),
                "corrected": str(it["corrected"]),
                "rule_key": str(it.get("rule_key") or kind),
                "explain_fr": str(it.get("explain_fr") or it.get("explanation") or ""),
            })
        self.reply.fixes = fixes

    def _parse_lesson(self) -> None:
        match = re.search(r"\{.*\}", self._lesson_buf, re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            data = None
        self.reply.lesson = data if isinstance(data, dict) else None


_EN_WORDS = frozenset(
    "the a an is are was were be been you your i i'm it it's this that what how why when where let's do does "
    "did don't can can't will would should have has had to of in on for with and but not we they he she".split())
_FR_WORDS = frozenset(
    "le la les un une des est sont était être tu te toi vous je j'ai c'est ce cette que qui quoi comment pourquoi "
    "quand où on dit de du et mais pas ne nous ils elle il au aux en avec pour sur très bien aussi essaie".split())


def guess_lang(text: str, default: str) -> str:
    """Langue d'une phrase non balisée, d'après ses mots-outils ; en cas de doute, la langue par défaut."""
    words = re.findall(r"[a-zà-ÿ']+", text.lower().replace("’", "'"))
    en = sum(w in _EN_WORDS for w in words)
    fr = sum(w in _FR_WORDS for w in words)
    if en - fr >= 2:
        return "en"
    if fr - en >= 2:
        return "fr"
    return default


_FIX_FIELD = re.compile(
    r'"(type|original|corrected|rule_key|explain_fr|explanation)"\s*:\s*"(.*?)"\s*(?=,\s*"\w+"\s*:|\s*$)', re.DOTALL)


def _repair_fixes(inner: str) -> list[dict]:
    """Relit des objets au JSON invalide, typiquement des guillemets non échappés dans
    explain_fr ("On dit "depend on""). Une valeur s'arrête au prochain champ ou à la fin de l'objet."""
    items = []
    for block in re.split(r"\}\s*,\s*\{", inner.strip().strip("{}")):
        fields = dict(_FIX_FIELD.findall(block.strip().strip("{}").strip()))
        if fields:
            items.append(fields)
    return items


def parse_full(text: str, default_lang: str = "en", fix_types: tuple[str, ...] = FIX_TYPES) -> ParsedReply:
    parser = ReplyParser(default_lang, fix_types)
    parser.feed(text)
    parser.close()
    return parser.reply
