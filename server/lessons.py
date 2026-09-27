"""Leçons : le programme écrit d'une matière (subjects/<matière>/lessons/*.yaml), ou une leçon générée par le
LLM sur un thème choisi par l'élève.

Le plan entre dans les consignes du prof, figées pour la séance (cache du LLM). Le prof signale où il en est
par une balise <lesson>{…}</lesson> entre </say> et <fix> ; le serveur suit l'avancement (partie en cours,
réponses au quiz), l'affiche, l'enregistre, et le rappelle au LLM en tête des messages de l'élève.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROGRAM = "programme:"
GENERATED = "gen:"


@dataclass(frozen=True)
class Lesson:
    key: str                     # "programme:01-organes" ou "gen:20260927-1430"
    title: str
    summary: str
    sections: tuple[dict, ...]   # {"title", "points": [...], "check"}
    quiz: tuple[dict, ...]       # {"question", "answer"}
    number: int | None = None    # rang dans le programme

    @classmethod
    def from_plan(cls, key: str, plan: dict, number: int | None = None) -> "Lesson":
        """Valide un plan (fichier du programme ou JSON produit par le LLM)."""
        sections = tuple(
            {"title": str(s["title"]).strip(), "points": [str(p).strip() for p in s.get("points") or []],
             "check": str(s.get("check") or "").strip()}
            for s in plan.get("sections") or [] if isinstance(s, dict) and s.get("title"))
        quiz = tuple({"question": str(q["question"]).strip(), "answer": str(q.get("answer") or "").strip()}
                     for q in plan.get("quiz") or [] if isinstance(q, dict) and q.get("question"))
        if not plan.get("title") or len(sections) < 2 or len(quiz) < 3:
            raise ValueError("plan de leçon incomplet : titre, au moins 2 parties et 3 questions de quiz")
        return cls(key, str(plan["title"]).strip(), str(plan.get("summary") or "").strip(), sections, quiz, number)

    @property
    def generated(self) -> bool:
        return self.key.startswith(GENERATED)

    def plan(self) -> dict:
        return {"title": self.title, "summary": self.summary, "sections": list(self.sections), "quiz": list(self.quiz)}

    def render(self) -> str:
        """Le plan tel que le prof le lit dans ses consignes."""
        lines = [f"# Plan de la leçon : {self.title}"]
        if self.summary:
            lines.append(self.summary)
        for i, s in enumerate(self.sections, 1):
            lines.append(f"\n## Partie {i} : {s['title']}")
            lines += [f"- {p}" for p in s["points"]]
            if s["check"]:
                lines.append(f"Question de compréhension : {s['check']}")
        lines.append(f"\n## Quiz ({len(self.quiz)} questions, dans l'ordre)")
        lines += [f"{i}. {q['question']} (réponse attendue : {q['answer']})" for i, q in enumerate(self.quiz, 1)]
        return "\n".join(lines)

    def public(self) -> dict:
        return {"key": self.key, "number": self.number, "title": self.title, "summary": self.summary,
                "sections": [s["title"] for s in self.sections], "questions": len(self.quiz),
                "generated": self.generated}


def load_program(directory: Path) -> tuple[Lesson, ...]:
    """Leçons du programme, dans l'ordre des noms de fichier (01-…, 02-…)."""
    if not directory.is_dir():
        return ()
    return tuple(
        Lesson.from_plan(PROGRAM + path.stem, yaml.safe_load(path.read_text(encoding="utf-8")), number)
        for number, path in enumerate(sorted(directory.glob("*.yaml")), 1))


def parse_generated(text: str) -> dict:
    """Plan JSON écrit par le LLM, éventuellement entouré de texte ou d'une clôture ```json."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("pas de JSON dans la réponse")
    return json.loads(match.group(0))


_STOPWORDS = frozenset("les des une est que qui pour dans par avec sur pas son ses aux leur leurs elle ils "
                       "elles cela ceci mais plus tres bien".split())


def sound(word: str) -> str:
    """Clé phonétique française sommaire : « poêles » et « poils » donnent la même (pwal)."""
    w = "".join(c for c in unicodedata.normalize("NFD", word.lower()) if unicodedata.category(c) != "Mn")
    for pattern, repl in ((r"eau|au", "o"), (r"o[ie]", "wa"), (r"ph", "f"), (r"ch", "x"), (r"qu", "k"),
                          (r"c(?=[eiy])", "s"), (r"c", "k"), (r"gn", "n"), (r"g(?=[eiy])", "j"), (r"h", ""),
                          (r"y", "i"), (r"[ae]i", "e"), (r"(.)\1", r"\1"), (r"s$", ""), (r"e$", ""),
                          (r"[tdxz]$", "")):
        w = re.sub(pattern, repl, w)
    return w


def homophones(said: str, expected: str) -> list[tuple[str, str]]:
    """Mots de la réponse transcrite qui se prononcent comme un mot de la réponse attendue sans s'écrire pareil :
    le STT a pu écrire « poêles absorbants » pour « poils absorbants », et le LLM ne l'entend pas."""
    def words(text: str) -> list[str]:
        return [w for w in re.findall(r"[^\W\d_]+", text.lower()) if len(w) >= 3 and w not in _STOPWORDS]
    targets = {sound(w): w for w in words(expected)}
    pairs = []
    for w in dict.fromkeys(words(said)):
        match = targets.get(sound(w))
        if match and match != w and len(sound(w)) >= 3:
            pairs.append((w, match))
    return pairs[:3]


def flatten(marker: dict) -> dict:
    """Gemma imbrique parfois la balise dans la précédente, {"section": {"question": 1}} au passage au quiz :
    on la remet à plat, sans quoi le quiz passerait inaperçu."""
    flat = {}
    for key, value in marker.items():
        if isinstance(value, dict):
            flat.update(flatten(value))
        else:
            flat.setdefault(key, value)
    return flat


@dataclass
class LessonState:
    """Où en est la leçon : partie en cours, question du quiz posée en dernier, réponses."""

    lesson: Lesson
    section: int = 1                 # partie en cours
    reached: int = 1                 # partie la plus avancée atteinte
    question: int = 0                # dernière question posée (0 : quiz pas commencé)
    results: dict[int, bool] = field(default_factory=dict)
    done: bool = False

    @property
    def score(self) -> int:
        return sum(self.results.values())

    def apply(self, marker: dict, said: str = "") -> bool:
        """Balise <lesson> d'une réponse du prof (said : ce qu'il a dit). Renvoie True si l'avancement a changé."""
        marker = flatten(marker)
        asked = re.findall(r"\bquestion\s+(\d+)\b", said, re.IGNORECASE)
        if asked and "question" not in marker and (self.question or self.reached == len(self.lesson.sections)):
            # vu en vrai : « Question 1 : … » annoncé avec {"section": 1} ; ce qu'il dit fait foi
            marker = {k: v for k, v in marker.items() if k != "section"} | {"question": int(asked[-1])}
        before = (self.section, self.reached, self.question, dict(self.results), self.done)
        n_sections, n_questions = len(self.lesson.sections), len(self.lesson.quiz)
        answered = marker.get("answered", self.question if "correct" in marker else 0)  # "correct" : ancien format
        verdict = marker.get("right", marker.get("correct"))
        if isinstance(answered, int) and 1 <= answered <= n_questions and verdict is not None and not self.done:
            self.results[answered] = bool(verdict)
        if isinstance(marker.get("section"), int) and not self.question:
            self.section = min(max(marker["section"], 1), n_sections)
            self.reached = max(self.reached, self.section)
        if isinstance(marker.get("question"), int) and not self.done:
            self.question = min(max(marker["question"], 1), n_questions)
            self.reached = n_sections
        if marker.get("done") or len(self.results) >= n_questions:
            self.done = self.done or bool(self.question)
        return before != (self.section, self.reached, self.question, self.results, self.done)

    def tag(self, minutes: int) -> str:
        """Repère ajouté en tête des messages de l'élève, pour que le prof tienne le fil et le rythme."""
        if self.done:
            return f"[Quiz terminé : {self.score}/{len(self.lesson.quiz)} · {minutes} min]"
        if self.question:
            n = len(self.lesson.quiz)
            last = " (la dernière)" if self.question == n else ""
            return (f"[Quiz : question {self.question}/{n} posée{last} · {self.score} bonne(s) réponse(s) sur "
                    f"{len(self.results)} · {minutes} min]")
        return f"[Leçon : partie {self.section}/{len(self.lesson.sections)} · {minutes} min]"

    def resume_point(self) -> str | None:
        """Où reprendre une leçon commencée lors d'une séance précédente."""
        if self.question:
            return f"la question {self.question} du quiz"
        if self.reached > 1:
            return f"la partie {self.reached} ({self.lesson.sections[self.reached - 1]['title']})"
        return None

    def progress(self) -> dict:
        return {"type": "lesson_progress", "section": self.section, "reached": self.reached,
                "sections": len(self.lesson.sections), "question": self.question,
                "questions": len(self.lesson.quiz), "results": {str(k): v for k, v in self.results.items()},
                "score": self.score, "done": self.done}

    def row(self) -> dict:
        return {"section": self.reached, "question": self.question, "done": self.done,
                "results": json.dumps({str(k): v for k, v in self.results.items()})}

    @classmethod
    def resumed(cls, lesson: Lesson, row) -> "LessonState":
        """Reprise d'une séance interrompue : même partie, même question, mêmes réponses."""
        results = {int(k): bool(v) for k, v in json.loads(row["results"] or "{}").items()}
        return cls(lesson, section=row["section"], reached=row["section"], question=row["question"],
                   results=results)
