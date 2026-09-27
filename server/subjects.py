"""Matières : un fichier YAML par prof dans subjects/ (nom, voix, langue, consignes, bilan)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from server.config import ROOT, resolve

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

    def system_prompt(self, memory_block: str, learner_name: str) -> str:
        """Figé pour toute la séance : c'est le préfixe que llama-server garde en cache."""
        learner = (f"# The learner\n- Name: {learner_name} (use it naturally, only if it is a real first name)\n"
                   + (memory_block or self.first_session))
        return self.prompt.replace("{teacher}", self.teacher).rstrip() + "\n\n" + learner

    def summary_prompt(self, transcript: str, mistakes: str, previous: str) -> str:
        return (self.summary.replace("{transcript}", transcript).replace("{mistakes}", mistakes)
                .replace("{previous}", previous))

    def public(self) -> dict:
        return {"id": self.id, "title": self.title, "teacher": self.teacher, "description": self.description,
                "ready": self.voice.exists()}


def load_subjects(directory: Path = SUBJECTS_DIR) -> dict[str, Subject]:
    subjects = {}
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["fix_types"] = tuple(raw["fix_types"])
        raw["voice"] = resolve(raw["voice"])
        subject = Subject(**raw)
        subjects[subject.id] = subject
    return subjects
