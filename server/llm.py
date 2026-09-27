"""llama-server (processus séparé, modèle résident) et client streaming."""

import asyncio
import json
import logging
import subprocess
import time
from collections.abc import AsyncIterator

import httpx
from huggingface_hub import hf_hub_download

from server.config import ROOT, resolve

log = logging.getLogger("profs.llm")


class LlamaServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.url = f"http://{cfg['host']}:{cfg['port']}"
        self.proc: subprocess.Popen | None = None

    def is_up(self) -> bool:
        try:
            return httpx.get(f"{self.url}/health", timeout=1).json().get("status") == "ok"
        except (httpx.HTTPError, ValueError):
            return False

    def start(self, timeout: float = 600) -> None:
        if self.is_up():
            log.info("llama-server déjà actif sur %s", self.url)
            return
        cfg = self.cfg
        model = hf_hub_download(cfg["model"]["repo"], cfg["model"]["file"])
        cmd = [str(resolve(cfg["server_bin"])), "-m", model, "--host", cfg["host"], "--port", str(cfg["port"]),
               "-c", str(cfg["ctx"]), *cfg["args"]]
        if cfg.get("draft"):
            cmd += ["--model-draft", hf_hub_download(cfg["draft"]["repo"], cfg["draft"]["file"])]
        log.info("Lancement : %s", " ".join(cmd))
        logfile = open(ROOT / "data" / "llama-server.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + timeout
        while not self.is_up():
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server s'est arrêté, voir data/llama-server.log")
            if time.monotonic() > deadline:
                raise TimeoutError("llama-server ne répond pas")
            time.sleep(0.5)
        log.info("llama-server prêt")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class ChatFormat:
    """Morceaux du modèle de chat, extraits une fois via /apply-template avec des marqueurs.

    On assemble ensuite nous-mêmes le prompt de chaque tour, en recopiant à l'identique les
    réponses déjà générées : le préfixe reste celui que llama-server a en cache. (Le modèle de
    chat de Gemma 4 retire des tours passés le canal de réflexion qu'il ajoute à la génération :
    via l'API chat, chaque tour re-calculait toute la réponse précédente.)
    """

    S, U, A, V = "⟦S⟧", "⟦U⟧", "⟦A⟧", "⟦V⟧"

    def __init__(self, rendered: str):
        s, u, a, v = (rendered.index(m) for m in (self.S, self.U, self.A, self.V))
        self.head = rendered[:u]                          # système + ouverture du 1er message élève
        self.after_user = rendered[v + len(self.V):]      # fin du message élève + ouverture de la réponse
        self.after_assistant = rendered[a + len(self.A): v]  # fin de réponse + ouverture du message élève suivant

    def render(self, system: str, history: list[dict], user: str) -> str:
        """history alterne user/assistant en commençant par user ; user est le nouveau message."""
        parts = [self.head.replace(self.S, system)]
        for m in history:
            parts += [m["content"], self.after_user if m["role"] == "user" else self.after_assistant]
        parts += [user, self.after_user]
        return "".join(parts)


class LLMClient:
    def __init__(self, cfg: dict):
        base = f"http://{cfg['host']}:{cfg['port']}"
        self.url = f"{base}/v1/chat/completions"
        self.completion_url = f"{base}/completion"
        self.template_url = f"{base}/apply-template"
        self.temperature = cfg.get("temperature", 0.6)
        self.max_tokens = cfg.get("max_tokens", 400)
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5))
        self.format: ChatFormat | None = None

    async def load_format(self) -> ChatFormat:
        f = ChatFormat
        msgs = [{"role": "system", "content": f.S}, {"role": "user", "content": f.U},
                {"role": "assistant", "content": f.A}, {"role": "user", "content": f.V}]
        resp = await self.http.post(self.template_url, json={"messages": msgs})
        resp.raise_for_status()
        self.format = ChatFormat(resp.json()["prompt"])
        return self.format

    async def stream_prompt(self, prompt: str, max_tokens: int | None = None, usage: dict | None = None,
                            stop: list[str] | None = None) -> AsyncIterator[str]:
        """Complétion brute d'un prompt déjà formaté. Fermer le générateur (aclose) coupe la
        connexion, ce qui arrête la génération côté llama-server. Si la génération va à son terme,
        usage["tokens"] reçoit le contexte occupé (prompt + réponse), compté par llama-server.
        Le mot d'arrêt rencontré (stop) est renvoyé en dernier : la sortie reste celle du modèle."""
        body = {"prompt": prompt, "stream": True, "temperature": self.temperature,
                "n_predict": max_tokens or self.max_tokens, "cache_prompt": True, "stop": stop or []}
        async with self.http.stream("POST", self.completion_url, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data.get("content"):
                    yield data["content"]
                if data.get("stop"):
                    if data.get("stop_type") == "word" and data.get("stopping_word"):
                        yield data["stopping_word"]
                    if usage is not None and "tokens_evaluated" in data:
                        usage["tokens"] = data["tokens_evaluated"] + data.get("tokens_predicted", 0)
                    return

    async def prime_prompt(self, prompt: str) -> None:
        try:
            await self.http.post(self.completion_url, json={"prompt": prompt, "n_predict": 0, "cache_prompt": True})
        except httpx.HTTPError as exc:
            log.warning("Préchauffage du cache LLM impossible : %s", exc)

    def _body(self, messages: list[dict], max_tokens: int | None, stream: bool) -> dict:
        return {
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "cache_prompt": True,
        }

    async def stream_chat(self, messages: list[dict], max_tokens: int | None = None) -> AsyncIterator[str]:
        """Génère la réponse morceau par morceau. Annuler la tâche ferme la connexion,
        ce qui arrête la génération côté llama-server."""
        async with self.http.stream("POST", self.url, json=self._body(messages, max_tokens, True)) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    return
                choices = json.loads(data).get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if delta:
                    yield delta

    async def complete(self, messages: list[dict], max_tokens: int | None = None) -> str:
        resp = await self.http.post(self.url, json=self._body(messages, max_tokens, False))
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def prime(self, messages: list[dict]) -> None:
        """Remplit le KV cache avec le préfixe (system prompt + mémoire) avant le premier tour."""
        try:
            await self.complete(messages, max_tokens=1)
        except httpx.HTTPError as exc:
            log.warning("Préchauffage du cache LLM impossible : %s", exc)

    async def aclose(self) -> None:
        await self.http.aclose()


async def _demo() -> None:  # python -m server.llm
    from server.config import load_config

    cfg = load_config()["llm"]
    server = LlamaServer(cfg)
    server.start()
    client = LLMClient(cfg)
    t0 = time.perf_counter()
    first = None
    async for delta in client.stream_chat([{"role": "user", "content": "Say hello in one short sentence."}]):
        first = first or time.perf_counter() - t0
        print(delta, end="", flush=True)
    print(f"\nTTFT {first * 1000:.0f} ms")
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(_demo())
