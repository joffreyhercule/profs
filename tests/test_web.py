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
