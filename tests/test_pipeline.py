"""Scénarios du pipeline avec de faux moteurs (sans GPU) : push-to-talk, tour spéculatif
confirmé ou annulé, interruption du prof."""

import asyncio
import json
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import server.pipeline as pipeline
from server.config import load_config
from server.llm import ChatFormat
from server.memory.db import MemoryDB
from server.pipeline import FRAME, Engines, Session
from server.subjects import load_subjects

REPLY = ('<say lang="en">Almost! We say I went, not I goed. Where did you go after school?</say>'
         '<fix>[{"type": "conjugation", "original": "I goed", "corrected": "I went", '
         '"rule_key": "past_simple_irregular", "explain_fr": "go est irrégulier"}]</fix>')


class FakeVAD:
    def __call__(self, frame):
        return 1.0 if np.abs(frame).mean() > 0.01 else 0.0


class FakeSTT:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return "yesterday I goed to school" if len(audio) else ""


class FakeTTS:
    def submit(self, job):
        if not job.cancel.is_set():
            job.on_chunk(b"\x01\x00" * 480)
            job.on_chunk(b"\x01\x00" * 480)
        job.on_done()


F = ChatFormat
FAKE_TEMPLATE = f"<sys>{F.S}</sys><user>{F.U}</user><model>{F.A}</model><user>{F.V}</user><model>"


class FakeLLM:
    """Répond REPLY ; si le prompt se termine déjà par un début de REPLY (reprise après pause),
    il continue à partir de là, comme llama-server."""

    def __init__(self):
        self.prompts = []
        self.format = ChatFormat(FAKE_TEMPLATE)
        self.used = 1000  # contexte occupé annoncé en fin de génération, comme llama-server

    async def stream_prompt(self, prompt, max_tokens=None, usage=None, stop=None):
        self.prompts.append(prompt)
        # Comme Gemma en fin de longue séance, le faux modèle ne s'arrête pas de lui-même après </fix> ;
        # llama-server coupe au mot d'arrêt, que le client renvoie en dernier.
        out = REPLY + "\n<fix>[]</fix>" * 3
        if stop:
            cut = min((out.find(w) for w in stop if w in out), default=-1)
            if cut >= 0:
                word = next(w for w in stop if out.startswith(w, cut))
                out = out[:cut + len(word)]
        done = next(k for k in range(len(out), -1, -1) if prompt.endswith(out[:k]))
        for i in range(done, len(out), 7):
            await asyncio.sleep(0)
            yield out[i: i + 7]
        if usage is not None:
            usage["tokens"] = self.used

    async def complete(self, messages, max_tokens=None):
        return '{"summary": "Bonne séance.", "level": "B1", "notes": "Practise past simple."}'

    async def prime_prompt(self, prompt):
        return 800  # taille du préfixe mis en cache, comme tokens_evaluated


class FakeSmartTurn:
    def __init__(self):
        self.p = 1.0

    def __call__(self, audio):
        return self.p


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_bytes(self, b):
        self.sent.append(b)

    async def send_text(self, t):
        self.sent.append(json.loads(t))

    def events(self, kind=None):
        evs = [m for m in self.sent if isinstance(m, dict)]
        return [e for e in evs if kind is None or e["type"] == kind]

    def audio(self):
        return [m for m in self.sent if isinstance(m, bytes)]


@pytest.fixture
async def setup(monkeypatch):
    monkeypatch.setattr(pipeline, "SileroVAD", FakeVAD)
    cfg = load_config()
    llm, smart = FakeLLM(), FakeSmartTurn()
    eng = Engines(cfg=cfg, stt=FakeSTT(), tts=FakeTTS(), smart_turn=smart, llm=llm,
                  db=MemoryDB(":memory:"), stt_executor=ThreadPoolExecutor(1), subjects=load_subjects())
    ws = FakeWS()
    session = Session(eng, ws, eng.db.create_user("Léa"), "anglais")
    # séance démarrée sans la salutation du prof, pour garder chaque scénario ciblé
    session.started = True
    session.session_id = eng.db.start_session(session.user_id, "anglais")
    writer = asyncio.create_task(session._writer())
    yield session, ws, eng
    writer.cancel()


def speech(ms):
    return (np.full(int(16000 * ms / 1000), 0.2) * 32767).astype("<i2").tobytes()


def silence(ms):
    return np.zeros(int(16000 * ms / 1000), "<i2").tobytes()


async def feed(session, data, realtime=False):
    """Envoie l'audio par blocs de 32 ms, en laissant tourner la boucle entre deux blocs."""
    step = FRAME * 2
    for i in range(0, len(data), step):
        session.on_audio(data[i: i + step])
        await asyncio.sleep(0.032 if realtime else 0.001)


async def settle(t=0.2):
    await asyncio.sleep(t)


async def test_push_to_talk_turn_is_answered_and_remembered(setup):
    session, ws, eng = setup
    await session.on_control({"type": "mode", "value": "ptt"})
    await session.on_control({"type": "ptt", "state": "down"})
    await feed(session, speech(600))
    await session.on_control({"type": "ptt", "state": "up"})
    await settle()

    assert ws.events("user_final")[0]["text"] == "yesterday I goed to school"
    assert [s["text"] for s in ws.events("segment")][:1] == ["Almost!"]
    assert ws.events("fixes")[0]["fixes"][0]["corrected"] == "I went"
    assert ws.audio() and struct.unpack("<II", ws.audio()[0][:8]) == (1, 0)
    assert ws.events("metrics")[0]["latency_ms"]["first_audio"] >= 0
    assert eng.db.top_errors(session.user_id, "anglais")[0]["rule_key"] == "past_simple_irregular"
    assert [m["role"] for m in session.history] == ["user", "assistant"]
    # le format est conservé pour les tours suivants, et rien de ce que le modèle écrirait après </fix>
    assert session.history[1]["content"] == REPLY


async def test_llm_pauses_after_first_segment_then_resumes_and_keeps_prefix(setup):
    session, ws, eng = setup
    for _ in range(2):
        await session.on_control({"type": "ptt", "state": "down"})
        await feed(session, speech(500))
        await session.on_control({"type": "ptt", "state": "up"})
        await settle()
        await session.on_control({"type": "played", "turn": session.live.id if session.live else 0})
    first, resumed, second = eng.llm.prompts[:3]
    assert resumed.startswith(first) and resumed != first  # reprise = prompt + début de la réponse
    assert second.startswith(first + REPLY)  # tour suivant : le texte en cache est rejoué à l'identique
    assert [s["text"] for s in ws.events("segment")][:3] == [
        "Almost!", "We say I went, not I goed.", "Where did you go after school?"]


async def test_eager_transcription_is_reused_when_key_released_after_silence(setup):
    session, ws, eng = setup
    await session.on_control({"type": "mode", "value": "ptt"})
    await session.on_control({"type": "ptt", "state": "down"})
    await feed(session, speech(500))
    await feed(session, silence(150))  # la transcription anticipée part pendant ce silence
    await settle(0.1)
    calls = eng.stt.calls
    await session.on_control({"type": "ptt", "state": "up"})
    await settle()
    assert eng.stt.calls == calls  # aucune transcription de plus sur le chemin critique
    assert ws.events("user_final")[0]["text"] == "yesterday I goed to school" and ws.audio()


async def test_speculative_turn_released_by_smart_turn(setup):
    session, ws, _ = setup
    await feed(session, speech(500))
    await feed(session, silence(400))
    await settle()
    assert ws.events("user_final") and ws.audio()


async def test_speculative_turn_cancelled_when_learner_continues(setup):
    session, ws, eng = setup
    eng.smart_turn.p = 0.0  # smart-turn pense que la phrase n'est pas finie
    await feed(session, speech(500))
    await feed(session, silence(300))  # tour spéculatif lancé, retenu
    await settle(0.1)
    assert session.spec is not None and not ws.events("user_final") and not ws.audio()
    await feed(session, speech(300))  # l'élève reprend : annulation
    assert session.spec is None
    await feed(session, silence(1700))  # silence long : fin de tour forcée
    await settle()
    assert len(ws.events("user_final")) == 1 and ws.audio()


async def test_barge_in_keeps_only_what_was_heard(setup):
    session, ws, _ = setup
    await session.on_control({"type": "ptt", "state": "down"})
    await feed(session, speech(500))
    await session.on_control({"type": "ptt", "state": "up"})
    await settle()
    turn = session.live
    await session.on_control({"type": "seg_start", "turn": turn.id, "seg": 0})
    await feed(session, speech(300))  # l'élève coupe le prof
    await settle(0.05)
    assert ws.events("flush")[0]["turn"] == turn.id
    assert session.history[-1]["content"].startswith('<say lang="en">Almost! [interrupted]</say>')
    assert session.live is None and session.utt is not None


async def test_end_session_writes_summary(setup):
    session, ws, eng = setup
    await session.on_control({"type": "ptt", "state": "down"})
    await feed(session, speech(500))
    await session.on_control({"type": "ptt", "state": "up"})
    await settle()
    await session.on_control({"type": "end_session"})
    await settle(0.05)
    assert ws.events("session_ended")[0]["summary"]["level"] == "B1"
    assert eng.db.get_profile(session.user_id, "anglais")["level"] == "B1"


async def test_history_is_halved_when_next_turn_would_not_fit_in_llm_context(setup):
    session, ws, eng = setup
    ctx = eng.cfg["llm"]["ctx"]
    sizes = []
    for used in (1000, 1000, ctx - 500):  # au 3e tour, llama-server annonce un contexte presque plein
        eng.llm.used = used
        await session.on_control({"type": "ptt", "state": "down"})
        await feed(session, speech(500))
        await session.on_control({"type": "ptt", "state": "up"})
        await settle()
        await session.on_control({"type": "played", "turn": session.live.id if session.live else 0})
        sizes.append(len(session.history))
    assert sizes == [2, 4, 2]  # les deux premiers échanges sont retirés d'un coup
    assert session.history[0]["role"] == "user" and not session.need_prime  # cache réchauffé après « played »
    # la page affiche le remplissage après chaque tour, puis la jauge redescend au réchauffage du cache
    await settle(0.05)
    gauge =[(e["used"], e["trims"]) for e in ws.events("context")]
    assert gauge == [(1000, 0), (1000, 0), (ctx - 500, 1), (800, 1)]
    assert all(e["max"] == ctx for e in ws.events("context"))
