"""Construit le bloc mémoire injecté dans le prompt et exploite le bilan de fin de séance."""

import json
import re

from server.memory.db import MemoryDB
from server.subjects import Subject


def memory_block(db: MemoryDB, user_id: int, subject: str, top_errors: int = 8) -> str:
    """Figé pour toute la séance (préfixe stable pour le KV cache du LLM)."""
    profile = db.get_profile(user_id, subject)
    last = db.last_finished_session(user_id, subject)
    errors = db.top_errors(user_id, subject, top_errors)
    vocab = db.vocabulary_to_review(user_id, subject, 8)
    if not (profile or last or errors):
        return ""

    lines = ["Memory of your previous sessions with this learner:"]
    if profile.get("level"):
        lines.append(f"- Estimated level: {profile['level']}. Adapt to this level.")
    if profile.get("notes"):
        lines.append(f"- Teacher notes: {profile['notes']}")
    if last is not None:
        lines.append(f"- Last session ({last['started_at'][:10]}), summary in French: {last['summary']}")
    if errors:
        lines.append("- Recurring mistakes or misconceptions (most frequent first). "
                     "Bring them up naturally and check they are fixed:")
        for e in errors:
            lines.append(
                f"  - {e['rule_key']} ({e['type']}, {e['count']}x over {e['sessions']} session(s)), "
                f'e.g. "{e["example_original"]}" -> "{e["example_corrected"]}"'
            )
    if vocab:
        words = ", ".join(v["word"] for v in vocab)
        lines.append(f"- Words or terms the learner got wrong before, reuse them: {words}")
    return "\n".join(lines)


def summary_prompt(db: MemoryDB, subject: Subject, session_id: int) -> str | None:
    turns = db.session_turns(session_id)
    if not any(t["role"] == "user" for t in turns):
        return None
    transcript = "\n".join(
        f"{'Learner' if t['role'] == 'user' else 'Teacher'}: {t['text']}" for t in turns
    )
    mistakes = "\n".join(
        f'- {e["rule_key"]}: "{e["original"]}" -> "{e["corrected"]}"' for e in db.session_errors(session_id)
    ) or "(none)"
    return subject.summary_prompt(transcript[-12000:], mistakes)


def apply_summary(db: MemoryDB, user_id: int, subject: str, session_id: int, llm_output: str) -> dict:
    """Lit le JSON produit par le LLM, clôt la séance et met à jour le profil de l'élève pour cette matière."""
    match = re.search(r"\{.*\}", llm_output, re.DOTALL)
    data = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = {}
    level = str(data.get("level") or "").strip()[:30] or None
    if level and re.fullmatch(r"[abcABC][12]", level):  # niveau CECRL
        level = level.upper()
    db.end_session(session_id, summary=data.get("summary"), level=level)
    db.set_profile(user_id, subject, {"level": level, "notes": data.get("notes")})
    return data
