"""Orchestration d'une séance (une connexion WebSocket).

Chemin critique : fin de parole -> STT final -> LLM en streaming -> segments -> TTS -> navigateur.
En mains libres, un tour « spéculatif » démarre dès min_silence_ms de silence : il génère
tout mais garde ses messages en réserve (held). Il est libéré dès que smart-turn confirme
la fin de phrase (ou au bout de max_silence_ms), et annulé si l'élève reprend la parole.
"""

import asyncio
import itertools
import json
import logging
import struct
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from server import health
from server.llm import LlamaServer, LLMClient
from server.memory.db import MemoryDB
from server.memory.profile import apply_summary, memory_block, summary_prompt
from server.output_parser import ParsedReply, ReplyParser, Segment
from server.stt import ParakeetSTT
from server.subjects import Subject
from server.tts import TeacherTTS, TTSJob
from server.vad import FRAME, SAMPLE_RATE, SileroVAD, SmartTurn

log = logging.getLogger("profs.pipeline")

FRAME_MS = FRAME * 1000 / SAMPLE_RATE
# Place gardée dans le contexte du LLM pour le message suivant de l'élève (un énoncé de 45 s fait
# ~200 tokens) et les balises du modèle de chat, en plus de la réponse (llm.max_tokens).
NEXT_TURN_TOKENS = 600
_background: set[asyncio.Task] = set()


def run_in_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


@dataclass
class Engines:
    cfg: dict
    stt: ParakeetSTT
    tts: TeacherTTS
    smart_turn: SmartTurn
    llm: LLMClient
    db: MemoryDB
    stt_executor: ThreadPoolExecutor
    llama: LlamaServer | None = None
    subjects: dict[str, Subject] = field(default_factory=dict)


class Turn:
    def __init__(self, session: "Session", turn_id: int, audio: np.ndarray | None, t_speech_end: float,
                 held: bool, user_text: str | None = None):
        self.s = session
        self.id = turn_id
        self.audio = audio
        self.user_text = user_text
        self.synthetic = user_text is not None
        self.held = held
        self.outbox: list = []
        self.released = asyncio.Event()
        if not held:
            self.released.set()
        self.task: asyncio.Task | None = None
        self.jobs: list[tuple[TTSJob, asyncio.Event]] = []
        self.segments: list[Segment] = []
        self.played: set[int] = set()
        self.reply: ParsedReply | None = None
        self.raw = ""  # sortie brute du LLM, balises comprises
        self.ctx_tokens: int | None = None  # contexte du LLM occupé en fin de tour, compté par llama-server
        self.first_chunk = asyncio.Event()  # premier morceau audio produit par le TTS
        self.stt_future: asyncio.Future | None = None
        self.cancelled = self.interrupted = self.committed = self.finished = False
        self.history_entry: dict | None = None
        self.db_assistant_id: int | None = None
        self.t: dict[str, float] = {"speech_end": t_speech_end}

    def mark(self, name: str) -> None:
        self.t.setdefault(name, time.perf_counter())

    def emit(self, msg) -> None:
        if self.cancelled:
            return
        if self.held:
            self.outbox.append(msg)
        else:
            self._send(msg)

    def _send(self, msg) -> None:
        if isinstance(msg, bytes):
            self.mark("first_audio")
        self.s.send(msg)

    def emit_audio(self, seg: int, pcm: bytes) -> None:
        self.first_chunk.set()
        self.emit(struct.pack("<II", self.id, seg) + pcm)

    def release(self) -> None:
        if not self.held or self.cancelled:
            return
        self.held = False
        self.mark("release")
        for msg in self.outbox:
            self._send(msg)
        self.outbox.clear()
        self.released.set()

    def stop_audio(self) -> None:
        for job, _ in self.jobs:
            job.cancel.set()

    def latencies(self) -> dict[str, int]:
        t0 = self.t["speech_end"]
        return {k: round((v - t0) * 1000) for k, v in self.t.items() if k != "speech_end"}


class Session:
    def __init__(self, eng: Engines, ws: WebSocket, user_id: int, subject: str):
        self.eng, self.ws, self.cfg = eng, ws, eng.cfg
        self.vcfg = eng.cfg["vad"]
        self.loop = asyncio.get_running_loop()
        self.out: asyncio.Queue = asyncio.Queue()
        self.db = eng.db
        self.user_id = user_id
        self.subject = eng.subjects[subject]
        self.session_id: int | None = None  # créée au démarrage de la séance, pas à la connexion
        mcfg = self.cfg["memory"]
        self.memory = memory_block(self.db, user_id, subject, mcfg["top_errors"], mcfg["recent_sessions"],
                                   mcfg["mastered_after_sessions"])
        self.system = self.subject.system_prompt(self.memory, self.db.get_user(user_id)["name"])
        self.history: list[dict] = []
        self.need_prime = False
        self.mode = "handsfree"
        self.vad = SileroVAD()
        self.turn_ids = itertools.count(1)
        self.live: Turn | None = None      # tour libéré (le prof pense ou parle)
        self.spec: Turn | None = None      # tour spéculatif de l'énoncé en cours
        self.utt: list[np.ndarray] | None = None
        self.utt_id = 0
        self.eager: asyncio.Future | None = None  # transcription anticipée de l'énoncé en cours
        self.preroll: deque[np.ndarray] = deque(maxlen=10)
        self.pending = np.zeros(0, np.float32)
        self.speech_run = self.silence_frames = 0
        self.last_voice_t = 0.0
        self.ptt = False
        self.partial_task: asyncio.Task | None = None
        self.last_partial_t = 0.0
        self.turn_check_t = 0.0
        self.turn_prob = 0.0  # dernier avis de smart-turn sur l'énoncé en cours
        self.started = self.ended = False

    # --- entrées/sorties -------------------------------------------------------------
    def send(self, msg) -> None:
        self.out.put_nowait(msg)

    async def _writer(self) -> None:
        while True:
            msg = await self.out.get()
            if isinstance(msg, bytes):
                await self.ws.send_bytes(msg)
            else:
                await self.ws.send_text(json.dumps(msg, ensure_ascii=False))

    async def run(self) -> None:
        writer = asyncio.create_task(self._writer())
        self.send({"type": "hello", "teacher": self.subject.teacher, "subject": self.subject.id,
                   "memory": self.memory})
        # prefill du system prompt + mémoire pendant que l'élève lit la page : la salutation part vite
        run_in_background(self.eng.llm.prime_prompt(self.prompt("")))
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    self.on_audio(msg["bytes"])
                elif msg.get("text"):
                    await self.on_control(json.loads(msg["text"]))
                if self.ended:
                    await asyncio.sleep(0.2)  # laisse partir les derniers messages
                    break
        except WebSocketDisconnect:
            pass
        finally:
            writer.cancel()
            if not self.ended:
                run_in_background(self.finish())

    async def on_control(self, msg: dict) -> None:
        kind = msg.get("type")
        now = time.perf_counter()
        if kind == "start" and not self.started:
            self.started = True
            self.session_id = self.db.start_session(self.user_id, self.subject.id)
            turn = self.launch(None, now, held=False, user_text=self.subject.greeting)
            self.live = turn
        elif kind == "mode":
            self.mode = msg.get("value", "handsfree")
            if self.mode != "handsfree" and not self.ptt:
                self.drop_utterance()
        elif kind == "ptt":
            if msg.get("state") == "down":
                self.ptt = True
                self.interrupt("ptt")
                self.cancel_spec()
                if self.utt is None:
                    self.start_utterance()
            elif self.ptt:
                self.ptt = False
                if self.utt is not None:
                    audio, eager = np.concatenate(self.utt), self.eager
                    self.drop_utterance()
                    self.live = self.launch(audio, now, held=False, stt=eager)
        elif kind == "stop":
            self.interrupt("button")
        elif kind == "seg_start":
            turn = self.live
            if turn and turn.id == msg.get("turn"):
                turn.played.add(int(msg.get("seg", -1)))
        elif kind == "played":
            turn = self.live
            if turn and turn.id == msg.get("turn") and turn.committed:
                turn.finished = True
                self.live = None
                if self.need_prime:
                    self.need_prime = False
                    run_in_background(self.eng.llm.prime_prompt(self.prompt("")))
        elif kind == "end_session":
            await self.finish()

    # --- audio et détection de tour -----------------------------------------------------
    def on_audio(self, data: bytes) -> None:
        if not self.started:
            return
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, samples])
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            self.on_frame(frame)

    def teacher_active(self) -> bool:
        return self.live is not None and not self.live.finished

    def on_frame(self, frame: np.ndarray) -> None:
        now = time.perf_counter()
        v = self.vcfg
        active = self.teacher_active()
        speech = self.vad(frame) >= (v["threshold_while_speaking"] if active else v["threshold"])
        self.speech_run = self.speech_run + 1 if speech else 0

        if self.utt is None:
            self.preroll.append(frame)
            if self.mode != "handsfree" or self.ended:
                return
            if self.speech_run >= (v["barge_in_frames"] if active else v["start_frames"]):
                self.interrupt("voice")
                self.start_utterance()
            return

        self.utt.append(frame)
        if speech:
            self.silence_frames = 0
            self.last_voice_t = now
            self.eager = None
            if self.spec is not None and self.speech_run >= v["start_frames"]:
                self.cancel_spec()  # l'élève n'avait pas fini
            self.maybe_partial(now)
            return
        self.silence_frames += 1
        if self.silence_frames == v["eager_stt_frames"] and self.eager is None:
            # transcription anticipée : prête avant la fin du silence d'attente ou le relâchement de la touche
            self.eager = self.loop.run_in_executor(self.eng.stt_executor, self.eng.stt.transcribe,
                                                   np.concatenate(self.utt))
        if self.ptt:
            return
        silence_ms = self.silence_frames * FRAME_MS
        if self.spec is None and silence_ms >= v["min_silence_ms"]:
            self.spec = self.launch(np.concatenate(self.utt), self.last_voice_t, held=True, stt=self.eager)
            self.turn_prob = 0.0
            self.turn_check_t = now
            run_in_background(self.check_end_of_turn(self.spec, self.spec.audio))
        elif self.spec is not None:
            # Libération seulement après min_release_ms de silence : le premier son n'est de toute façon
            # pas prêt avant, et une hésitation plus courte ne fait pas parler le prof par-dessus l'élève.
            finished = self.turn_prob >= self.cfg["turn"]["threshold"] and silence_ms >= v["min_release_ms"]
            if finished or silence_ms >= v["max_silence_ms"]:
                self.release_spec()
            elif (now - self.turn_check_t) * 1000 >= v["recheck_ms"]:
                self.turn_check_t = now
                run_in_background(self.check_end_of_turn(self.spec, np.concatenate(self.utt)))

    async def check_end_of_turn(self, turn: Turn, audio: np.ndarray) -> None:
        prob = await asyncio.to_thread(self.eng.smart_turn, audio)
        if turn is self.spec and not turn.cancelled:
            self.turn_prob = prob

    def start_utterance(self) -> None:
        self.utt = list(self.preroll)
        self.utt_id += 1
        self.eager = None
        self.silence_frames = 0
        self.last_voice_t = time.perf_counter()
        self.last_partial_t = time.perf_counter()
        self.send({"type": "listening", "utt": self.utt_id})

    def drop_utterance(self) -> None:
        self.utt = None
        self.eager = None
        self.preroll.clear()
        self.speech_run = self.silence_frames = 0

    def maybe_partial(self, now: float) -> None:
        if (now - self.last_partial_t) * 1000 < self.cfg["stt"]["partial_interval_ms"]:
            return
        if self.partial_task and not self.partial_task.done():
            return
        self.last_partial_t = now
        self.partial_task = asyncio.create_task(self.partial(self.utt_id, np.concatenate(self.utt)))

    async def partial(self, utt_id: int, audio: np.ndarray) -> None:
        text = await self.loop.run_in_executor(self.eng.stt_executor, self.eng.stt.transcribe, audio)
        if utt_id == self.utt_id and self.utt is not None and text:
            self.send({"type": "partial", "utt": utt_id, "text": text})

    def release_spec(self) -> None:
        turn, self.spec = self.spec, None
        if turn is None or turn.cancelled:
            return
        self.drop_utterance()
        self.live = turn
        turn.release()

    def cancel_spec(self) -> None:
        turn, self.spec = self.spec, None
        if turn is None:
            return
        turn.cancelled = True
        turn.stop_audio()
        if turn.task:
            turn.task.cancel()

    def interrupt(self, reason: str) -> None:
        """Coupe le prof : vidage du lecteur, arrêt LLM + TTS, l'historique garde ce qui a été entendu."""
        turn = self.live
        if turn is None or turn.finished:
            return
        turn.interrupted = True
        self.send({"type": "flush", "turn": turn.id, "reason": reason})
        turn.stop_audio()
        if turn.task and not turn.task.done():
            turn.task.cancel()
        if turn.committed:
            self.patch_interrupted(turn)
        else:
            self.commit(turn)
        turn.finished = turn.cancelled = True  # plus aucun message de ce tour
        self.live = None

    # --- un tour de réponse ----------------------------------------------------------
    def launch(self, audio: np.ndarray | None, t_speech_end: float, held: bool,
               user_text: str | None = None, stt: asyncio.Future | None = None) -> Turn:
        turn = Turn(self, next(self.turn_ids), audio, t_speech_end, held, user_text)
        turn.stt_future = stt
        turn.task = asyncio.create_task(self.run_turn(turn))
        return turn

    def prompt(self, user_text: str) -> str:
        """Prompt complet du tour : identique, caractère pour caractère, au texte déjà en cache
        dans llama-server (prompt précédent + réponse générée), suivi du nouveau message."""
        history = self.history
        if history and history[-1]["role"] == "user":  # le prof n'avait rien dit : on fusionne
            user_text = history[-1]["content"] + "\n" + user_text
            history = history[:-1]
        return self.eng.llm.format.render(self.system, history, user_text)

    async def generate(self, turn: Turn, parser: ReplyParser, prompt: str, pause_at_first_segment: bool) -> bool:
        """Fait avancer la réponse. Renvoie True si on s'est arrêté après le premier segment
        (le LLM libère le GPU pour que la voix produise son premier son plus vite)."""
        usage: dict = {}
        # Arrêt net après </fix> : passé ~7 000 tokens de contexte, Gemma enchaînait des « <fix>[]</fix> »
        # jusqu'à max_tokens (3,5 s de génération contre la voix : trous audio), et l'historique, qui garde
        # la sortie brute, entretenait la dérive (mesuré : 17 réponses sur 120).
        stream = self.eng.llm.stream_prompt(prompt, usage=usage, stop=["</fix>"])
        try:
            async for delta in stream:
                turn.mark("llm_first_token")
                turn.raw += delta
                segments, display = parser.feed(delta)
                if display:
                    turn.emit({"type": "assistant_delta", "turn": turn.id, "text": display})
                for seg in segments:
                    self.speak(turn, seg)
                if pause_at_first_segment and turn.segments and parser.state != "done":
                    return True
        finally:
            await stream.aclose()  # coupe la connexion : llama-server arrête de générer
            turn.ctx_tokens = usage.get("tokens", turn.ctx_tokens)
        return False

    async def run_turn(self, turn: Turn) -> None:
        try:
            if turn.user_text is None:
                if turn.stt_future is not None:  # transcription anticipée (partagée : on la protège de l'annulation)
                    text = await asyncio.shield(turn.stt_future)
                else:
                    text = await self.loop.run_in_executor(self.eng.stt_executor, self.eng.stt.transcribe, turn.audio)
                turn.mark("stt")
                if not any(c.isalnum() for c in text):
                    await turn.released.wait()
                    turn.emit({"type": "noinput", "turn": turn.id})
                    turn.finished = True
                    if self.live is turn:
                        self.live = None
                    return
                turn.user_text = text
                turn.emit({"type": "user_final", "turn": turn.id, "text": text})

            parser = ReplyParser(self.subject.lang, self.subject.fix_types)
            turn.reply = parser.reply  # rempli au fil du flux (langue dès <say>, corrections à la fin)
            turn.emit({"type": "assistant_start", "turn": turn.id})
            prompt = self.prompt(turn.user_text)
            lcfg = self.cfg["llm"]
            if await self.generate(turn, parser, prompt, lcfg.get("pause_after_first_segment", True)):
                try:
                    await asyncio.wait_for(turn.first_chunk.wait(), lcfg.get("pause_max_ms", 250) / 1000)
                except TimeoutError:
                    pass
                turn.mark("llm_resume")
                await self.generate(turn, parser, prompt + turn.raw, pause_at_first_segment=False)
            segments, display = parser.close()
            if display:
                turn.emit({"type": "assistant_delta", "turn": turn.id, "text": display})
            for seg in segments:
                self.speak(turn, seg)
            turn.reply = parser.reply
            turn.mark("llm_done")
            turn.emit({"type": "fixes", "turn": turn.id, "user_text": "" if turn.synthetic else turn.user_text,
                       "fixes": turn.reply.fixes, "parse_error": turn.reply.fix_parse_error})
            await asyncio.gather(*(done.wait() for _, done in turn.jobs))
            turn.mark("tts_done")
            turn.emit({"type": "assistant_done", "turn": turn.id, "segments": len(turn.segments)})
            await turn.released.wait()
            self.commit(turn)
            if not turn.segments:
                turn.finished = True
                if self.live is turn:
                    self.live = None
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.exception("Échec du tour %s", turn.id)
            llama = self.eng.llama
            if health.is_gpu_fatal(exc) or (
                    isinstance(exc, httpx.HTTPError) and llama is not None and not await asyncio.to_thread(llama.is_up)):
                health.fatal(f"tour {turn.id}")
            turn.emit({"type": "error", "turn": turn.id, "message": "Le tour a échoué, voir les logs du serveur."})
            turn.finished = True
            if self.live is turn:
                self.live = None

    def speak(self, turn: Turn, seg: Segment) -> None:
        seg_id = len(turn.segments)
        turn.segments.append(seg)
        turn.mark("first_segment")
        done = asyncio.Event()
        job = TTSJob(
            seg.text, seg.lang,
            on_chunk=lambda pcm: self.loop.call_soon_threadsafe(turn.emit_audio, seg_id, pcm),
            on_done=lambda: self.loop.call_soon_threadsafe(done.set),
            voice=self.subject.id,
        )
        turn.jobs.append((job, done))
        turn.emit({"type": "segment", "turn": turn.id, "seg": seg_id, "text": seg.text, "lang": seg.lang})
        self.eng.tts.submit(job)

    # --- mémoire ---------------------------------------------------------------------
    def spoken_text(self, turn: Turn) -> str:
        return " ".join(turn.segments[i].text for i in sorted(turn.played) if i < len(turn.segments))

    def commit(self, turn: Turn) -> None:
        if turn.committed or turn.cancelled or not turn.user_text:
            return
        turn.committed = True
        full = turn.reply.say_text if turn.reply else ""
        said = self.spoken_text(turn) if turn.interrupted else full
        self._append("user", turn.user_text)
        if said:
            # le LLM relit ses réponses passées : elles doivent garder le format <say>/<fix>,
            # sinon il l'imite et l'abandonne
            content = self.interrupted_reply(turn, said) if turn.interrupted else (turn.raw or said)
            turn.history_entry = self._append("assistant", content)
        user_turn_id = None if turn.synthetic else self.db.add_turn(self.session_id, "user", turn.user_text)
        if turn.reply and turn.reply.fixes and not turn.synthetic:
            self.db.add_errors(self.user_id, self.subject.id, self.session_id, user_turn_id, turn.reply.fixes)
        if said:
            turn.db_assistant_id = self.db.add_turn(self.session_id, "assistant", said, interrupted=turn.interrupted)
        lat = turn.latencies()
        self.db.add_metrics(self.session_id, {"turn": turn.id, **lat})
        self.send({"type": "metrics", "turn": turn.id, "latency_ms": lat})
        self.trim_history(turn.ctx_tokens)

    @staticmethod
    def interrupted_reply(turn: Turn, said: str) -> str:
        lang = turn.reply.lang if turn.reply else "en"
        fixes = json.dumps(turn.reply.fixes if turn.reply else [], ensure_ascii=False)
        return f'<say lang="{lang}">{said} [interrupted]</say>\n<fix>{fixes}</fix>'

    def patch_interrupted(self, turn: Turn) -> None:
        said = self.spoken_text(turn)
        if turn.history_entry is not None and turn.history_entry in self.history:
            if said:
                turn.history_entry["content"] = self.interrupted_reply(turn, said)
            else:
                self.history.remove(turn.history_entry)
        if turn.db_assistant_id is not None:
            self.db.update_turn(turn.db_assistant_id, said, interrupted=True)

    def _append(self, role: str, content: str) -> dict:
        if self.history and self.history[-1]["role"] == role:
            self.history[-1]["content"] += "\n" + content
            return self.history[-1]
        entry = {"role": role, "content": content}
        self.history.append(entry)
        return entry

    def trim_history(self, used: int | None) -> None:
        """Si le tour suivant risque de ne plus tenir dans le contexte du LLM, on retire d'un coup la moitié
        la plus ancienne de l'historique (le préfixe change rarement), puis on réchauffe le cache quand le
        prof a fini de parler. used : contexte occupé à la fin de ce tour, compté par llama-server."""
        lcfg = self.cfg["llm"]
        if used is None:  # tour interrompu avant la fin : estimation prudente (~3 caractères par token)
            used = (len(self.system) + sum(len(m["content"]) for m in self.history)) // 3
        if used + NEXT_TURN_TOKENS + lcfg["max_tokens"] <= lcfg["ctx"]:
            return
        cut = len(self.history) // 2
        while cut < len(self.history) and self.history[cut]["role"] != "user":
            cut += 1
        del self.history[:cut]
        self.need_prime = True

    async def finish(self) -> None:
        """Fin de séance : résumé par le LLM et mise à jour du profil."""
        if self.ended:
            return
        self.ended = True
        self.interrupt("end")
        self.cancel_spec()
        summary = {}
        if self.session_id is None:  # page quittée avant de commencer
            return
        prompt = summary_prompt(self.db, self.subject, self.user_id, self.session_id)
        if prompt:
            try:
                out = await self.eng.llm.complete([{"role": "user", "content": prompt}], max_tokens=400)
                summary = apply_summary(self.db, self.user_id, self.subject.id, self.session_id, out)
            except Exception:
                log.exception("Résumé de séance impossible")
                self.db.end_session(self.session_id)
        else:
            self.db.delete_session(self.session_id)  # l'élève n'a rien dit : rien à retenir
        self.send({"type": "session_ended", "session_id": self.session_id, "summary": summary})
