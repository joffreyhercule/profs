"""Matières : un fichier YAML par prof dans subjects/ (nom, voix, langue, consignes, bilan).
Une matière peut aussi donner des leçons : consignes dans son bloc `lesson`, programme dans
subjects/<matière>/lessons/*.yaml (voir server/lessons.py)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from server.config import ROOT, resolve
from server.lessons import Lesson, load_program

SUBJECTS_DIR = ROOT / "subjects"


@dataclass(frozen=True)
class Subject:
    id: str
    title: str
    teacher: str
    description: str
    lang: str
    fix_types: tuple[str, ...]
    voice: Path
    voice_language: str
    voice_description: str
    voice_sample: str
    prompt: str
    first_session: str
    greeting: str
    summary: str
    lesson: dict | None = None          # consignes des séances-leçons : prompt, greeting, resume, generate
    lessons: tuple[Lesson, ...] = ()    # programme écrit

    def system_prompt(self, memory_block: str, learner_name: str, lesson: Lesson | None = None) -> str:
        """Figé pour toute la séance : c'est le préfixe que llama-server garde en cache."""
        learner = (f"# The learner\n- Name: {learner_name} (use it naturally, only if it is a real first name)\n"
                   + (memory_block or self.first_session))
        prompt = self.prompt.replace("{teacher}", self.teacher).rstrip()
        if lesson is not None and self.lesson:
            prompt += "\n\n" + self.lesson["prompt"].replace("{plan}", lesson.render()).rstrip()
        return prompt + "\n\n" + learner

    def find_lesson(self, key: str) -> Lesson | None:
        return next((lesson for lesson in self.lessons if lesson.key == key), None)

    def summary_prompt(self, transcript: str, mistakes: str, previous: str) -> str:
        return (self.summary.replace("{transcript}", transcript).replace("{mistakes}", mistakes)
                .replace("{previous}", previous))

    def public(self) -> dict:
        return {"id": self.id, "title": self.title, "teacher": self.teacher, "description": self.description,
                "ready": self.voice.exists(), "lessons": bool(self.lesson)}


def load_subjects(directory: Path = SUBJECTS_DIR) -> dict[str, Subject]:
    subjects = {}
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["fix_types"] = tuple(raw["fix_types"])
        raw["voice"] = resolve(raw["voice"])
        raw["lessons"] = load_program(directory / raw["id"] / "lessons")
        subject = Subject(**raw)
        subjects[subject.id] = subject
    return subjects
