import sqlite3

from server.memory.db import LEGACY_USER_NAME, MemoryDB
from server.memory.profile import apply_summary, memory_block, summary_prompt
from server.subjects import load_subjects

FIX = {"type": "conjugation", "original": "I goed", "corrected": "I went",
       "rule_key": "past_simple_irregular", "explain_fr": "go est irrégulier"}
SUBJECTS = load_subjects()
EN = "anglais"


def test_first_session_has_empty_memory():
    db = MemoryDB(":memory:")
    uid = db.create_user("Léa")
    sid = db.start_session(uid, EN)
    assert memory_block(db, uid, EN) == ""
    assert summary_prompt(db, SUBJECTS[EN], sid) is None


def test_errors_accumulate_across_sessions_and_feed_next_prompt():
    db = MemoryDB(":memory:")
    uid = db.create_user("Léa")
    s1 = db.start_session(uid, EN)
    t1 = db.add_turn(s1, "user", "yesterday I goed to school")
    db.add_turn(s1, "assistant", "We say I went.")
    db.add_errors(uid, EN, s1, t1, [FIX])
    db.add_errors(uid, EN, s1, t1, [FIX])
    prompt = summary_prompt(db, SUBJECTS[EN], s1)
    assert "Learner: yesterday I goed to school" in prompt and "past_simple_irregular" in prompt
    apply_summary(db, uid, EN, s1, 'Voici : {"summary": "On a parlé de l\'école.", "level": "b1", "notes": "Likes football."}')

    s2 = db.start_session(uid, EN)
    db.add_errors(uid, EN, s2, None, [FIX])
    stats = db.top_errors(uid, EN)[0]
    assert (stats["count"], stats["sessions"]) == (3, 2)

    block = memory_block(db, uid, EN)
    assert "Estimated level: B1" in block
    assert "Likes football." in block
    assert "On a parlé de l'école." in block
    assert '"I goed" -> "I went"' in block


def test_memory_is_separate_per_learner_and_per_subject():
    db = MemoryDB(":memory:")
    lea, tom = db.create_user("Léa"), db.create_user("Tom")
    db.add_errors(lea, EN, db.start_session(lea, EN), None, [FIX])
    assert db.top_errors(tom, EN) == []
    assert db.top_errors(lea, "botanique") == []
    assert memory_block(db, tom, EN) == "" and memory_block(db, lea, "botanique") == ""
    assert [r["name"] for r in db.users()] == ["Léa", "Tom"]  # le plus récemment actif d'abord


def test_subject_prompt_names_teacher_and_learner():
    prompt = SUBJECTS["botanique"].system_prompt("", "Léa")
    assert prompt.startswith("Tu es Basile") and "Name: Léa" in prompt
    assert "premier cours" in prompt  # pas de mémoire : consignes de première séance


def test_free_text_level_is_kept_for_non_language_subjects():
    db = MemoryDB(":memory:")
    uid = db.create_user("Léa")
    sid = db.start_session(uid, "botanique")
    db.add_turn(sid, "user", "c'est quoi la photosynthèse ?")
    apply_summary(db, uid, "botanique", sid, '{"summary": "Photosynthèse.", "level": "débutant", "notes": "Revoir."}')
    assert db.get_profile(uid, "botanique")["level"] == "débutant"


def test_vocabulary_mistakes_are_tracked():
    db = MemoryDB(":memory:")
    uid = db.create_user("Léa")
    db.add_errors(uid, EN, db.start_session(uid, EN), None, [
        {"type": "false_friend", "original": "actually", "corrected": "currently",
         "rule_key": "false_friend_actually", "explain_fr": ""}])
    assert db.vocabulary_to_review(uid, EN)[0]["word"] == "currently"


def test_session_without_learner_speech_is_forgotten():
    db = MemoryDB(":memory:")
    uid = db.create_user("Léa")
    sid = db.start_session(uid, EN)
    db.add_turn(sid, "assistant", "Hello!")
    db.delete_session(sid)
    assert db.sessions(uid, EN) == [] and db.session_turns(sid) == []


def test_invalid_summary_still_closes_session():
    db = MemoryDB(":memory:")
    uid = db.create_user("Léa")
    sid = db.start_session(uid, EN)
    apply_summary(db, uid, EN, sid, "pas de json")
    assert db.sessions(uid, EN)[0]["ended_at"] is not None


LEGACY_SCHEMA = """
CREATE TABLE sessions (id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, summary TEXT, level TEXT);
CREATE TABLE turns (id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL,
                    lang TEXT, interrupted INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE errors (id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, turn_id INTEGER, type TEXT NOT NULL,
                     original TEXT NOT NULL, corrected TEXT NOT NULL, rule_key TEXT NOT NULL, explain_fr TEXT,
                     created_at TEXT NOT NULL);
CREATE TABLE error_stats (rule_key TEXT PRIMARY KEY, type TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0,
                          sessions INTEGER NOT NULL DEFAULT 0, last_session_id INTEGER, last_seen TEXT,
                          example_original TEXT, example_corrected TEXT, explain_fr TEXT);
CREATE TABLE vocabulary (word TEXT PRIMARY KEY, wrong_form TEXT, times_wrong INTEGER NOT NULL DEFAULT 0,
                         first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);
CREATE TABLE learner_profile (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE metrics (id INTEGER PRIMARY KEY, session_id INTEGER, data TEXT NOT NULL, created_at TEXT NOT NULL);
INSERT INTO sessions VALUES (1, '2026-09-26T14:52:51', '2026-09-26T14:56:44', 'Présentations.', 'A2');
INSERT INTO error_stats VALUES ('third_person_s', 'conjugation', 3, 1, 1, '2026-09-26', 'he have', 'he has', '-s');
INSERT INTO vocabulary VALUES ('currently', 'actually', 1, '2026-09-26', '2026-09-26');
INSERT INTO learner_profile VALUES ('level', 'A2', '2026-09-26');
"""


def test_legacy_database_is_migrated_to_a_first_profile(tmp_path):
    path = tmp_path / "profs.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(LEGACY_SCHEMA)
    db = MemoryDB(path)
    (user,) = db.users()
    assert user["name"] == LEGACY_USER_NAME and user["n_sessions"] == 1
    assert db.get_profile(user["id"], EN) == {"level": "A2"}
    assert db.top_errors(user["id"], EN)[0]["count"] == 3
    assert db.vocabulary_to_review(user["id"], EN)[0]["word"] == "currently"
    assert "Présentations." in memory_block(db, user["id"], EN)
    assert (tmp_path / "profs.v0-backup.db").exists()
    MemoryDB(path)  # une seconde ouverture ne remigre pas
    assert len(MemoryDB(path).users()) == 1
