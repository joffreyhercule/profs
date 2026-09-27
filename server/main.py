"""Point d'entrée : python -m server.main

Charge et préchauffe tous les modèles une seule fois (ils restent résidents),
puis sert la page web et le WebSocket /ws.
"""

import logging
import os
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server import health
from server.config import ROOT, load_config, resolve
from server.llm import LlamaServer, LLMClient
from server.memory.db import MemoryDB
from server.pipeline import Engines, Session
from server.stt import ParakeetSTT
from server.subjects import load_subjects
from server.tts import TeacherTTS
from server.vad import SmartTurn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("profs")

CFG = load_config()
state: dict = {}


def vram_used_gb() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        return info.used / 1024**3
    except Exception:
        return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t0 = time.perf_counter()
    (ROOT / "data").mkdir(exist_ok=True)
    llama = LlamaServer(CFG["llm"])
    llama.start()  # en premier : c'est lui qui réserve le plus de VRAM
    health.on_fatal(llama.stop)
    log.info("VRAM après LLM : %.1f Go", vram_used_gb() or -1)
    stt = ParakeetSTT(CFG["stt"])
    stt.warmup()
    subjects = load_subjects()
    tts = TeacherTTS(CFG["tts"], {s.id: s.voice for s in subjects.values()})
    tts.warmup()
    llm = LLMClient(CFG["llm"])
    fmt = await llm.load_format()
    async for _ in llm.stream_prompt(fmt.render("You are helpful.", [], "Hi"), max_tokens=8):
        pass  # premier passage : graphes CUDA de llama-server
    state["engines"] = Engines(
        cfg=CFG, stt=stt, tts=tts, smart_turn=SmartTurn(CFG["turn"]), llm=llm,
        # PROFS_DB : base séparée pour les bancs d'essai, qui ne doivent pas polluer la mémoire de l'élève
        db=MemoryDB(resolve(os.environ.get("PROFS_DB") or CFG["memory"]["db"])),
        stt_executor=ThreadPoolExecutor(1, thread_name_prefix="stt"),
        llama=llama,
        # une matière sans voix (design_voice.py pas encore lancé) n'est pas proposée
        subjects={k: s for k, s in subjects.items() if k in tts.voices},
    )
    log.info("Prêt en %.0f s, VRAM utilisée : %.1f Go", time.perf_counter() - t0, vram_used_gb() or -1)
    yield
    await llm.aclose()
    llama.stop()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, user: int, subject: str) -> None:
    eng = state["engines"]
    if eng.db.get_user(user) is None or subject not in eng.subjects:
        await ws.close(code=4404)
        return
    await ws.accept()
    await Session(eng, ws, user, subject).run()


class UserIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)


@app.get("/api/subjects")
def subjects() -> list[dict]:
    return [s.public() for s in state["engines"].subjects.values()]


@app.get("/api/users")
def users() -> list[dict]:
    return [dict(r) for r in state["engines"].db.users()]


@app.post("/api/users")
def create_user(body: UserIn) -> dict:
    db = state["engines"].db
    return dict(db.get_user(db.create_user(body.name.strip())))


@app.patch("/api/users/{user_id}")
def rename_user(user_id: int, body: UserIn) -> dict:
    db = state["engines"].db
    if db.get_user(user_id) is None:
        raise HTTPException(404, "Profil inconnu")
    db.rename_user(user_id, body.name.strip())
    return dict(db.get_user(user_id))


@app.get("/api/sessions")
def sessions(user: int, subject: str) -> list[dict]:
    return [dict(r) for r in state["engines"].db.sessions(user, subject)]


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: int) -> dict:
    db = state["engines"].db
    return {"turns": [dict(r) for r in db.session_turns(session_id)],
            "errors": [dict(r) for r in db.session_errors(session_id)]}


@app.get("/api/stats")
def stats(user: int, subject: str) -> dict:
    db = state["engines"].db
    mastered_after = CFG["memory"]["mastered_after_sessions"]
    return {"top_errors": [dict(r) for r in db.top_errors(user, subject, 15, mastered_after)],
            "profile": db.get_profile(user, subject), "vram_gb": vram_used_gb()}


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")


def main() -> None:
    host, port = CFG["server"]["host"], CFG["server"]["port"]
    config = uvicorn.Config(app, host=host, port=port, log_level="info", ws_max_size=2**20)
    server = uvicorn.Server(config)
    orig_startup = server.startup

    async def startup(sockets=None):
        await orig_startup(sockets)
        if not os.environ.get("PROFS_NO_BROWSER"):
            webbrowser.open(f"http://{host}:{port}")

    server.startup = startup
    server.run()


if __name__ == "__main__":
    main()
