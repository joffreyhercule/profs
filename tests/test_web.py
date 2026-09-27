"""La page : ses fichiers sont versionnés par empreinte, pour que le navigateur ne garde pas une vieille version."""

import shutil

from fastapi.testclient import TestClient

import server.main as main


def test_page_references_versioned_assets_and_is_always_revalidated():
    client = TestClient(main.app)  # sans lifespan : aucun modèle chargé
    page = client.get("/")
    version = main.web_version()
    assert page.headers["cache-control"] == "no-cache"
    assert f'src="app.js?v={version}"' in page.text and f'href="styles.css?v={version}"' in page.text
    asset = client.get(f"/app.js?v={version}")
    assert asset.status_code == 200 and "immutable" in asset.headers["cache-control"]
    assert client.get("/app.js").headers["cache-control"] == "no-cache"


def test_version_changes_when_a_page_file_changes(tmp_path, monkeypatch):
    web = tmp_path / "web"
    shutil.copytree(main.WEB, web)
    monkeypatch.setattr(main, "WEB", web)
    before = main.web_version()
    (web / "player-worklet.js").write_text("// modifié", encoding="utf-8")
    assert main.web_version() != before


def test_lessons_api_gives_progress_through_the_program(monkeypatch):
    from types import SimpleNamespace

    from server.lessons import Lesson, LessonState
    from server.memory.db import MemoryDB
    from server.subjects import load_subjects

    subjects, db = load_subjects(), MemoryDB(":memory:")
    monkeypatch.setitem(main.state, "engines", SimpleNamespace(db=db, subjects=subjects))
    uid = db.create_user("Léa")
    organes, photo, _ = subjects["botanique"].lessons
    done = LessonState(organes, question=10, results={i: i <= 8 for i in range(1, 11)}, done=True)
    started = LessonState(photo, section=3, reached=3)
    generated = LessonState(Lesson.from_plan("gen:1", {**organes.plan(), "title": "Les lichens"}), done=True,
                            question=10, results={1: True})
    for state in (done, started, generated):
        sid = db.start_session(uid, "botanique")
        db.start_lesson_run(sid, uid, "botanique", state.lesson.key, state.lesson.title, state.lesson.plan(),
                            state.row())

    body = TestClient(main.app).get(f"/api/lessons?user={uid}&subject=botanique").json()
    assert [(p["number"], p["status"]) for p in body["program"]] == [(1, "done"), (2, "started"), (3, "todo")]
    assert body["program"][0]["score"] == "8/10" and body["program"][1]["resume"] == "partie 3"
    assert body["next"] == "programme:02-photosynthese"
    assert [g["title"] for g in body["generated"]] == ["Les lichens"]
    assert TestClient(main.app).get(f"/api/lessons?user={uid}&subject=anglais").json() == {"enabled": False}
