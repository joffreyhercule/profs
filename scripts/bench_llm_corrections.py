"""Banc LLM : le prof détecte-t-il les fautes attendues, sans surcorriger, dans un format lisible ?
Mesure aussi le premier token et le premier segment prêt pour le TTS.

Usage : python scripts/bench_llm_corrections.py [--repeat 2] [--limit N]
(llama-server est lancé avec le modèle de config.yaml s'il ne tourne pas déjà)
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import ROOT, load_config  # noqa: E402
from server.llm import LlamaServer, LLMClient  # noqa: E402
from server.output_parser import ReplyParser  # noqa: E402
from server.subjects import load_subjects  # noqa: E402


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w' ]+", " ", s.lower().replace("’", "'")).split())


def pct(values: list[float], q: float) -> float:
    return statistics.quantiles(values, n=100)[int(q) - 1] if len(values) > 1 else values[0]


async def run(repeat: int, limit: int | None) -> None:
    cfg = load_config()
    LlamaServer(cfg["llm"]).start()
    client = LLMClient(cfg["llm"])
    system = load_subjects()["anglais"].system_prompt("", "Test")
    data = yaml.safe_load((ROOT / "tests" / "data" / "sentences.yaml").read_text(encoding="utf-8"))
    await client.prime([{"role": "system", "content": system}, {"role": "user", "content": "Hi"}])

    rows, ttft, first_seg = [], [], []
    for kind in ("wrong", "correct"):
        for item in data[kind][:limit]:
            for _ in range(repeat):
                parser = ReplyParser()
                t0 = time.perf_counter()
                t_tok = t_seg = None
                raw = ""
                async for delta in client.stream_chat([{"role": "system", "content": system},
                                                       {"role": "user", "content": item["text"]}]):
                    t_tok = t_tok or time.perf_counter() - t0
                    raw += delta
                    segs, _ = parser.feed(delta)
                    if segs and t_seg is None:
                        t_seg = time.perf_counter() - t0
                segs, _ = parser.close()
                if segs and t_seg is None:
                    t_seg = time.perf_counter() - t0
                reply = parser.reply
                ttft.append(t_tok * 1000)
                if t_seg:
                    first_seg.append(t_seg * 1000)
                fixes = reply.fixes
                if kind == "wrong":
                    ok = any(norm(item["error"]) in norm(f["original"]) or norm(item["expected"]) in norm(f["corrected"])
                             for f in fixes)
                else:
                    ok = not fixes
                fmt_ok = bool(reply.segments) and not reply.fix_parse_error and "<fix>" in raw
                rows.append({"id": item["id"], "kind": kind, "ok": ok, "format_ok": fmt_ok,
                             "ttft_ms": round(t_tok * 1000), "text": item["text"], "raw": raw})
                mark = "OK " if ok else "RATÉ"
                print(f"{mark} {item['id']} {'' if fmt_ok else '[format] '}{reply.say_text[:90]!r} "
                      f"fixes={[(f['original'], f['corrected']) for f in fixes]}", flush=True)

    wrong = [r for r in rows if r["kind"] == "wrong"]
    correct = [r for r in rows if r["kind"] == "correct"]
    summary = {
        "model": cfg["llm"]["model"],
        "detection": f"{sum(r['ok'] for r in wrong)}/{len(wrong)}",
        "sans_surcorrection": f"{sum(r['ok'] for r in correct)}/{len(correct)}",
        "format_ok": f"{sum(r['format_ok'] for r in rows)}/{len(rows)}",
        "ttft_ms_p50": round(statistics.median(ttft)), "ttft_ms_p95": round(pct(ttft, 95)),
        "premier_segment_ms_p50": round(statistics.median(first_seg)) if first_seg else None,
        "premier_segment_ms_p95": round(pct(first_seg, 95)) if first_seg else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = ROOT / "bench_results"
    out.mkdir(exist_ok=True)
    path = out / f"llm_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Détail : {path}")
    await client.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(args.repeat, args.limit))
