"""Erreurs GPU irrécupérables (reset du pilote par Windows, « TDR ») : après un TDR, tous les
contextes CUDA sont perdus. On sort avec le code RESTART_CODE et run.bat relance tout."""

import logging
import os
from collections.abc import Callable

log = logging.getLogger("profs.health")

RESTART_CODE = 3
_on_exit: list[Callable[[], None]] = []


def on_fatal(callback: Callable[[], None]) -> None:
    _on_exit.append(callback)


def is_gpu_fatal(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("cuda error", "cuda failure", "cudaerror", "device-side assert",
                                   "illegal memory access", "unspecified launch failure"))


def fatal(reason: str) -> None:
    log.critical("Erreur GPU irrécupérable (%s) : redémarrage complet du prof", reason)
    for callback in _on_exit:
        try:
            callback()
        except Exception:
            log.exception("Échec pendant l'arrêt")
    os._exit(RESTART_CODE)
