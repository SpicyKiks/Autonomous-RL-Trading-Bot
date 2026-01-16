"""Live trading session manager for dashboard control."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from autonomous_rl_trading_bot.common.paths import artifacts_dir


class SessionStatus(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


@dataclass
class LiveSession:
    """Singleton live trading session state."""

    status: SessionStatus = SessionStatus.IDLE
    thread: Optional[threading.Thread] = None
    stop_event: Optional[threading.Event] = None
    run_id: Optional[str] = None
    run_dir: Optional[Path] = None
    last_error: Optional[str] = None
    log_lines: list[str] = None

    def __post_init__(self):
        if self.log_lines is None:
            self.log_lines = []

    def reset(self):
        """Reset session to idle state."""
        self.status = SessionStatus.IDLE
        self.thread = None
        self.stop_event = None
        self.run_id = None
        self.run_dir = None
        self.last_error = None
        self.log_lines = []


# Global singleton instance
_LIVE_SESSION = LiveSession()


def get_session() -> LiveSession:
    """Get the global live trading session."""
    return _LIVE_SESSION


def start_trading(args_list: list[str], run_id: str, run_dir: Path) -> None:
    """Start live trading in background thread."""
    session = get_session()

    if session.status == SessionStatus.RUNNING:
        raise RuntimeError("Trading session already running. Stop it first.")

    session.status = SessionStatus.RUNNING
    session.run_id = run_id
    session.run_dir = run_dir
    session.last_error = None
    session.stop_event = threading.Event()
    session.log_lines = []

    def _run_trading():
        """Background thread function."""
        logger = logging.getLogger("arbt")
        try:
            # Import here to avoid circular imports
            import run_live_demo

            # Redirect logging to capture log lines
            class LogCaptureHandler(logging.Handler):
                def __init__(self, session_ref):
                    super().__init__()
                    self.session_ref = session_ref

                def emit(self, record):
                    msg = self.format(record)
                    self.session_ref.log_lines.append(msg)
                    # Keep only last 200 lines
                    if len(self.session_ref.log_lines) > 200:
                        self.session_ref.log_lines = self.session_ref.log_lines[-200:]

            handler = LogCaptureHandler(session)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(handler)

            try:
                # Run live trading
                run_live_demo.main(args_list)
            finally:
                logger.removeHandler(handler)

            session.status = SessionStatus.IDLE
        except Exception as e:
            session.status = SessionStatus.ERROR
            session.last_error = str(e)
            logger.error(f"Live trading error: {e}", exc_info=True)

    session.thread = threading.Thread(target=_run_trading, daemon=True, name="LiveTradingThread")
    session.thread.start()


def stop_trading() -> None:
    """Stop live trading session."""
    session = get_session()

    if session.status != SessionStatus.RUNNING:
        return

    session.status = SessionStatus.STOPPING

    # Signal stop via kill switch file (use global kill switch path)
    kill_switch_path = artifacts_dir() / "KILL_SWITCH"
    try:
        kill_switch_path.write_text("STOP", encoding="utf-8")
    except Exception:
        pass

    # Signal stop event
    if session.stop_event:
        session.stop_event.set()

    # Wait for thread to finish (with timeout)
    if session.thread and session.thread.is_alive():
        session.thread.join(timeout=5.0)

    session.status = SessionStatus.IDLE
