"""Stockage SQLite des élèves, séances, tours, erreurs et profils.

La mémoire (erreurs récurrentes, vocabulaire, niveau, notes) est propre à chaque couple
élève × matière : ce qu'un élève rate en anglais ne pollue pas son cours de botanique.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1
LEGACY_USER_NAME = "Mon profil"
LEGACY_SUBJECT = "anglais"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    subject TEXT NOT NULL DEFAULT 'anglais',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    summary TEXT,
    level TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    lang TEXT,
    interrupted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    turn_id INTEGER REFERENCES turns(id),
    type TEXT NOT NULL,
    original TEXT NOT NULL,
    corrected TEXT NOT NULL,
    rule_key TEXT NOT NULL,
    explain_fr TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS error_stats (
    user_id INTEGER NOT NULL REFERENCES users(id),
    subject TEXT NOT NULL,
    rule_key TEXT NOT NULL,
    type TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    sessions INTEGER NOT NULL DEFAULT 0,
    last_session_id INTEGER,
    last_seen TEXT,
    example_original TEXT,
    example_corrected TEXT,
    explain_fr TEXT,
    PRIMARY KEY (user_id, subject, rule_key)
);
CREATE TABLE IF NOT EXISTS vocabulary (
    user_id INTEGER NOT NULL REFERENCES users(id),
    subject TEXT NOT NULL,
    word TEXT NOT NULL,
    wrong_form TEXT,
    times_wrong INTEGER NOT NULL DEFAULT 0,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    PRIMARY KEY (user_id, subject, word)
);
CREATE TABLE IF NOT EXISTS learner_profile (
    user_id INTEGER NOT NULL REFERENCES users(id),
    subject TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, subject, key)
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY,
    session_id INTEGER,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lesson_runs (
    session_id INTEGER PRIMARY KEY REFERENCES sessions(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    subject TEXT NOT NULL,
    lesson_key TEXT NOT NULL,        -- "programme:01-organes" ou "gen:…" (voir server/lessons.py)
    title TEXT NOT NULL,
    plan TEXT NOT NULL,              -- plan JSON : seule copie des leçons générées
    section INTEGER NOT NULL DEFAULT 1,   -- partie la plus avancée atteinte
    question INTEGER NOT NULL DEFAULT 0,  -- dernière question du quiz posée (0 : pas commencé)
    results TEXT NOT NULL DEFAULT '{}',   -- {"1": true, "2": false, …}
    done INTEGER NOT NULL DEFAULT 0,      -- quiz terminé
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id, subject);
CREATE INDEX IF NOT EXISTS lesson_runs_user ON lesson_runs(user_id, subject, lesson_key);
"""

# Types de correction qui portent sur un mot ou un terme à retenir
VOCAB_TYPES = ("vocabulary", "false_friend", "terminology")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class MemoryDB:
    def __init__(self, path: Path | str):
        self.path = None if str(path) == ":memory:" else Path(path)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        legacy = version == 0 and self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sessions'").fetchone()
        if legacy:
            self._migrate_legacy()
        self.conn.executescript(SCHEMA)
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def _migrate_legacy(self) -> None:
        """Base d'avant les profils et les matières : toute la mémoire existante est rattachée à un
        premier élève, en anglais. Une copie de l'ancienne base est gardée à côté (*.v0-backup.db)."""
        c = self.conn
        if self.path:
            with sqlite3.connect(self.path.with_name(f"{self.path.stem}.v0-backup.db")) as backup:
                c.backup(backup)
        c.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL)")
        uid = c.execute("INSERT INTO users(name, created_at) VALUES (?, ?)", (LEGACY_USER_NAME, _now())).lastrowid
        c.execute("ALTER TABLE sessions ADD COLUMN user_id INTEGER REFERENCES users(id)")
        c.execute(f"ALTER TABLE sessions ADD COLUMN subject TEXT NOT NULL DEFAULT '{LEGACY_SUBJECT}'")
        c.execute("UPDATE sessions SET user_id = ?", (uid,))
        for table in ("error_stats", "vocabulary", "learner_profile"):
            c.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
        c.executescript(SCHEMA)
        c.execute("""INSERT INTO error_stats SELECT ?, ?, rule_key, type, count, sessions, last_session_id, last_seen,
                         example_original, example_corrected, explain_fr FROM error_stats_legacy""",
                  (uid, LEGACY_SUBJECT))
        c.execute("""INSERT INTO vocabulary SELECT ?, ?, word, wrong_form, times_wrong, first_seen, last_seen
                     FROM vocabulary_legacy""", (uid, LEGACY_SUBJECT))
        c.execute("INSERT INTO learner_profile SELECT ?, ?, key, value, updated_at FROM learner_profile_legacy",
                  (uid, LEGACY_SUBJECT))
        for table in ("error_stats", "vocabulary", "learner_profile"):
            c.execute(f"DROP TABLE {table}_legacy")
        c.commit()

    # --- élèves --------------------------------------------------------------------------
    def create_user(self, name: str) -> int:
        cur = self.conn.execute("INSERT INTO users(name, created_at) VALUES (?, ?)", (name, _now()))
        self.conn.commit()
        return cur.lastrowid

    def rename_user(self, user_id: int, name: str) -> None:
        self.conn.execute("UPDATE users SET name = ? WHERE id = ?", (name, user_id))
        self.conn.commit()

    def get_user(self, user_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    def users(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT u.*, COUNT(s.id) AS n_sessions, MAX(s.started_at) AS last_session
               FROM users u LEFT JOIN sessions s ON s.user_id = u.id
               GROUP BY u.id ORDER BY last_session IS NULL, last_session DESC, u.id"""
        ).fetchall()

    # --- séances -----------------------------------------------------------------------------
    def start_session(self, user_id: int, subject: str) -> int:
        cur = self.conn.execute("INSERT INTO sessions(user_id, subject, started_at) VALUES (?, ?, ?)",
                                (user_id, subject, _now()))
        self.conn.commit()
        return cur.lastrowid

    def end_session(self, session_id: int, summary: str | None = None, level: str | None = None) -> None:
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?, summary = COALESCE(?, summary), level = COALESCE(?, level) WHERE id = ?",
            (_now(), summary, level, session_id),
        )
        self.conn.commit()

    def delete_session(self, session_id: int) -> None:
        """Séance ouverte puis quittée sans que l'élève ait parlé : rien à retenir."""
        for table in ("errors", "turns", "metrics", "lesson_runs"):
            self.conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()

    def recent_summaries(self, user_id: int, subject: str, limit: int = 3) -> list[sqlite3.Row]:
        """Dernières séances résumées, la plus récente d'abord."""
        return self.conn.execute(
            """SELECT * FROM sessions WHERE user_id = ? AND subject = ? AND summary IS NOT NULL
               ORDER BY id DESC LIMIT ?""",
            (user_id, subject, limit),
        ).fetchall()

    def sessions(self, user_id: int, subject: str, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT s.*, (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id AND t.role = 'user') AS n_turns,
                      (SELECT COUNT(*) FROM errors e WHERE e.session_id = s.id) AS n_errors
               FROM sessions s WHERE s.user_id = ? AND s.subject = ? ORDER BY s.id DESC LIMIT ?""",
            (user_id, subject, limit),
        ).fetchall()

    # --- tours et erreurs ----------------------------------------------------
    def add_turn(self, session_id: int, role: str, text: str, lang: str | None = None,
                 interrupted: bool = False) -> int:
        cur = self.conn.execute(
            "INSERT INTO turns(session_id, role, text, lang, interrupted, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, text, lang, int(interrupted), _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_turn(self, turn_id: int, text: str, interrupted: bool) -> None:
        self.conn.execute("UPDATE turns SET text = ?, interrupted = ? WHERE id = ?", (text, int(interrupted), turn_id))
        self.conn.commit()

    def session_turns(self, session_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()

    def add_errors(self, user_id: int, subject: str, session_id: int, turn_id: int | None,
                   fixes: list[dict]) -> None:
        now = _now()
        for f in fixes:
            self.conn.execute(
                """INSERT INTO errors(session_id, turn_id, type, original, corrected, rule_key, explain_fr, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, turn_id, f["type"], f["original"], f["corrected"], f["rule_key"], f.get("explain_fr"), now),
            )
            self.conn.execute(
                """INSERT INTO error_stats(user_id, subject, rule_key, type, count, sessions, last_session_id,
                                           last_seen, example_original, example_corrected, explain_fr)
                   VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, subject, rule_key) DO UPDATE SET
                       count = count + 1,
                       sessions = sessions + (last_session_id IS NOT excluded.last_session_id),
                       last_session_id = excluded.last_session_id,
                       last_seen = excluded.last_seen,
                       example_original = excluded.example_original,
                       example_corrected = excluded.example_corrected,
                       explain_fr = excluded.explain_fr""",
                (user_id, subject, f["rule_key"], f["type"], session_id, now, f["original"], f["corrected"],
                 f.get("explain_fr")),
            )
            if f["type"] in VOCAB_TYPES:
                self.conn.execute(
                    """INSERT INTO vocabulary(user_id, subject, word, wrong_form, times_wrong, first_seen, last_seen)
                       VALUES (?, ?, ?, ?, 1, ?, ?)
                       ON CONFLICT(user_id, subject, word) DO UPDATE SET times_wrong = times_wrong + 1,
                           wrong_form = excluded.wrong_form, last_seen = excluded.last_seen""",
                    (user_id, subject, f["corrected"].lower(), f["original"], now, now),
                )
        self.conn.commit()

    def session_errors(self, session_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM errors WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()

    # Séances terminées depuis la dernière fois que l'élève a raté la notion (ou le terme). À partir de
    # mastered_after, elle est considérée comme acquise ; si l'élève la rate de nouveau, elle revient.
    _ERROR_CLEAN_SESSIONS = """(SELECT COUNT(*) FROM sessions s WHERE s.user_id = e.user_id AND s.subject = e.subject
                                AND s.ended_at IS NOT NULL AND s.id > e.last_session_id)"""
    _VOCAB_CLEAN_SESSIONS = """(SELECT COUNT(*) FROM sessions s WHERE s.user_id = v.user_id AND s.subject = v.subject
                                AND s.ended_at IS NOT NULL AND s.started_at > v.last_seen)"""

    def top_errors(self, user_id: int, subject: str, limit: int = 8,
                   mastered_after: int | None = None) -> list[sqlite3.Row]:
        """Erreurs récurrentes ; avec mastered_after, sans celles qui sont acquises."""
        not_mastered = f"AND {self._ERROR_CLEAN_SESSIONS} < {int(mastered_after)}" if mastered_after else ""
        return self.conn.execute(
            f"""SELECT e.* FROM error_stats e WHERE e.user_id = ? AND e.subject = ? {not_mastered}
                ORDER BY e.sessions DESC, e.count DESC, e.last_seen DESC LIMIT ?""",
            (user_id, subject, limit),
        ).fetchall()

    def mastered_errors(self, user_id: int, subject: str, mastered_after: int, limit: int = 5) -> list[sqlite3.Row]:
        """Erreurs acquises, les plus récemment ratées d'abord."""
        return self.conn.execute(
            f"""SELECT e.* FROM error_stats e WHERE e.user_id = ? AND e.subject = ?
                AND {self._ERROR_CLEAN_SESSIONS} >= ? ORDER BY e.last_seen DESC LIMIT ?""",
            (user_id, subject, mastered_after, limit),
        ).fetchall()

    def vocabulary_to_review(self, user_id: int, subject: str, limit: int = 10,
                             mastered_after: int | None = None) -> list[sqlite3.Row]:
        not_mastered = f"AND {self._VOCAB_CLEAN_SESSIONS} < {int(mastered_after)}" if mastered_after else ""
        return self.conn.execute(
            f"""SELECT v.* FROM vocabulary v WHERE v.user_id = ? AND v.subject = ? {not_mastered}
                ORDER BY v.times_wrong DESC, v.last_seen DESC LIMIT ?""",
            (user_id, subject, limit),
        ).fetchall()

    # --- leçons -------------------------------------------------------------------
    def start_lesson_run(self, session_id: int, user_id: int, subject: str, key: str, title: str,
                         plan: dict, progress: dict) -> None:
        self.conn.execute(
            """INSERT INTO lesson_runs(session_id, user_id, subject, lesson_key, title, plan, section, question,
                                       results, done, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, user_id, subject, key, title, json.dumps(plan, ensure_ascii=False), progress["section"],
             progress["question"], progress["results"], int(progress["done"]), _now()))
        self.conn.commit()

    def update_lesson_run(self, session_id: int, progress: dict) -> None:
        self.conn.execute(
            "UPDATE lesson_runs SET section = ?, question = ?, results = ?, done = ?, updated_at = ? WHERE session_id = ?",
            (progress["section"], progress["question"], progress["results"], int(progress["done"]), _now(),
             session_id))
        self.conn.commit()

    def lesson_run(self, session_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM lesson_runs WHERE session_id = ?", (session_id,)).fetchone()

    def last_lesson_run(self, user_id: int, subject: str, key: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM lesson_runs WHERE user_id = ? AND subject = ? AND lesson_key = ? ORDER BY session_id DESC",
            (user_id, subject, key)).fetchone()

    def lesson_runs(self, user_id: int, subject: str) -> list[sqlite3.Row]:
        """Dernière séance de chaque leçon suivie par l'élève, la plus récente d'abord, et s'il l'a déjà finie."""
        return self.conn.execute(
            """SELECT r.*, (SELECT MAX(done) FROM lesson_runs d WHERE d.user_id = r.user_id AND d.subject = r.subject
                            AND d.lesson_key = r.lesson_key) AS ever_done
               FROM lesson_runs r
               WHERE r.user_id = ? AND r.subject = ? AND r.session_id = (
                   SELECT MAX(session_id) FROM lesson_runs l WHERE l.user_id = r.user_id AND l.subject = r.subject
                   AND l.lesson_key = r.lesson_key)
               ORDER BY r.session_id DESC""",
            (user_id, subject)).fetchall()

    # --- profil et métriques ----------------------------------------------------
    def get_profile(self, user_id: int, subject: str) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM learner_profile WHERE user_id = ? AND subject = ?",
                                 (user_id, subject))
        return {r["key"]: r["value"] for r in rows}

    def set_profile(self, user_id: int, subject: str, values: dict[str, str]) -> None:
        now = _now()
        self.conn.executemany(
            "INSERT INTO learner_profile(user_id, subject, key, value, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, subject, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            [(user_id, subject, k, v, now) for k, v in values.items() if v],
        )
        self.conn.commit()

    def add_metrics(self, session_id: int | None, data: dict) -> None:
        self.conn.execute(
            "INSERT INTO metrics(session_id, data, created_at) VALUES (?, ?, ?)",
            (session_id, json.dumps(data), _now()),
        )
        self.conn.commit()
