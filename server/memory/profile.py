"""Construit le bloc mémoire injecté dans le prompt et exploite le bilan de fin de séance."""

import json
import re

from server.memory.db import MemoryDB
from server.subjects import Subject


def memory_block(db: MemoryDB, user_id: int, subject: str, top_errors: int = 8, recent_sessions: int = 3,
                 mastered_after: int = 3) -> str:
    """Figé pour toute la séance (préfixe stable pour le KV cache du LLM)."""
    profile = db.get_profile(user_id, subject)
    recent = db.recent_summaries(user_id, subject, recent_sessions)
    errors = db.top_errors(user_id, subject, top_errors, mastered_after)
    mastered = db.mastered_errors(user_id, subject, mastered_after)
    vocab = db.vocabulary_to_review(user_id, subject, 8, mastered_after)
    lessons = db.lesson_runs(user_id, subject)[:4]
    if not (profile or recent or errors or mastered or lessons):
        return ""

    lines = ["Memory of your previous sessions with this learner:"]
    if profile.get("level"):
        lines.append(f"- Estimated level: {profile['level']}. Adapt to this level.")
    if profile.get("notes"):
        lines.append(f"- Teacher notes: {profile['notes']}")
    if recent:
        lines.append("- Recent sessions (most recent first), summaries in French:")
        lines += [f"  - {s['started_at'][:10]}: {s['summary']}" for s in recent]
    if lessons:
        lines.append("- Lessons followed (most recent first): " + "; ".join(lesson_status(r) for r in lessons))
    if errors:
        lines.append("- Recurring mistakes or misconceptions (most frequent first). "
                     "Bring them up naturally and check they are fixed:")
        for e in errors:
            lines.append(
                f"  - {e['rule_key']} ({e['type']}, {e['count']}x over {e['sessions']} session(s)), "
                f'e.g. "{e["example_original"]}" -> "{e["example_corrected"]}"'
            )
    if mastered:
        lines.append(f"- Mastered: not repeated in the last {mastered_after} sessions, no need to insist "
                     f"(praise it briefly if it comes up): {', '.join(e['rule_key'] for e in mastered)}")
    if vocab:
        words = ", ".join(v["word"] for v in vocab)
        lines.append(f"- Words or terms the learner got wrong before, reuse them: {words}")
    return "\n".join(lines)


def lesson_status(run) -> str:
    """« Titre » et où l'élève en est, d'après une ligne de lesson_runs."""
    results = json.loads(run["results"] or "{}")
    n_questions = len(json.loads(run["plan"]).get("quiz", []))
    if run["done"]:
        state = f"finished, quiz {sum(results.values())}/{n_questions}"
    elif run["question"]:
        state = f"stopped at quiz question {run['question']}"
    else:
        state = f"stopped at part {run['section']}"
    return f"« {run['title']} » ({state})"


def summary_prompt(db: MemoryDB, subject: Subject, user_id: int, session_id: int) -> str | None:
    turns = db.session_turns(session_id)
    if not any(t["role"] == "user" for t in turns):
        return None
    transcript = "\n".join(
        f"{'Learner' if t['role'] == 'user' else 'Teacher'}: {t['text']}" for t in turns
    )
    run = db.lesson_run(session_id)
    if run is not None:
        transcript = f"(Lesson: {lesson_status(run)})\n" + transcript
    mistakes = "\n".join(
        f'- {e["rule_key"]}: "{e["original"]}" -> "{e["corrected"]}"' for e in db.session_errors(session_id)
    ) or "(none)"
    # niveau et notes d'avant ce cours : le LLM les met à jour au lieu de repartir de zéro
    profile = db.get_profile(user_id, subject.id)
    previous = "\n".join(f"- {k}: {profile[k]}" for k in ("level", "notes") if profile.get(k)) or "(none)"
    return subject.summary_prompt(transcript[-12000:], mistakes, previous)


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
