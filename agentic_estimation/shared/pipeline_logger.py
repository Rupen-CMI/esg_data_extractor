"""
pipeline_logger.py — shared logging setup for the ESG agentic pipeline.

Usage in each agent:
    from agentic_estimation.shared.pipeline_logger import get_logger, log_header
    log = get_logger("scoring_agent")
    log_header("Scoring Agent", company="Bosch", extra="Industrial Machinery")
"""

import logging
import os
from pathlib import Path

_LOGS_DIR = Path(__file__).parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)

_FMT = "%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s"
_DATE = "%H:%M:%S"

_PIPELINE_LOG = _LOGS_DIR / "pipeline.log"

# One shared file handler writing everything to pipeline.log
_file_handler: logging.FileHandler | None = None


def _get_file_handler() -> logging.FileHandler:
    global _file_handler
    if _file_handler is None:
        _file_handler = logging.FileHandler(_PIPELINE_LOG, encoding="utf-8")
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
    return _file_handler


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that writes ONLY to the log files (logs/pipeline.log +
    logs/<name>.log), never to the console — the pipeline runs inside the API
    server, so its output must not pollute the server's stdout.
    Safe to call multiple times — handlers are not duplicated.
    """
    log = logging.getLogger(name)
    if log.handlers:
        return log  # already configured

    log.setLevel(logging.DEBUG)

    # Shared file handler (no console handler — file-only by design)
    log.addHandler(_get_file_handler())

    # Also write to an agent-specific log file
    agent_log = _LOGS_DIR / f"{name}.log"
    fh = logging.FileHandler(agent_log, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(fh)

    log.propagate = False
    return log


def log_pipeline_start(company: str, mode: str = "dry") -> None:
    """
    Write a highly visible separator to pipeline.log marking the start of a new run.
    Includes blank lines above so old and new runs are easy to tell apart in the log file.
    """
    import datetime
    log = get_logger("orchestrator")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    width = 54
    sep = "#" * width
    log.info(
        "\n\n\n"
        "%s\n"
        "##  NEW PIPELINE RUN\n"
        "##  company : %s\n"
        "##  mode    : %s\n"
        "##  started : %s\n"
        "%s",
        sep, company, mode.upper(), ts, sep,
    )


def log_header(log: logging.Logger, agent_name: str, **kwargs) -> None:
    """
    Emit a prominent banner to mark the start of an agent run.

    Example output:
    ╔══════════════════════════════════════════╗
    ║  SIGNAL AGENT                            ║
    ║  company : Bosch                         ║
    ║  industry: Industrial Machinery          ║
    ╚══════════════════════════════════════════╝
    """
    width = 46
    lines = [agent_name.upper()]
    for k, v in kwargs.items():
        lines.append(f"{k:<9}: {v}")

    top    = "=" * width
    bottom = "=" * width
    body   = "\n".join(f"  {l}" for l in lines)

    banner = f"\n+{top}+\n{body}\n+{bottom}+"
    log.info(banner)
