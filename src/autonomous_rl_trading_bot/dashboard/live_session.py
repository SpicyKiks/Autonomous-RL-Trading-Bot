"""Live trading session manager for dashboard control."""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from autonomous_rl_trading_bot.common.paths import artifacts_dir


class SessionStatus(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    DONE = "DONE"
    ERROR = "ERROR"


@dataclass
class LiveSession:
    """Singleton live trading session state."""

    status: SessionStatus = SessionStatus.IDLE
    thread: threading.Thread | None = None
    stop_event: threading.Event | None = None
    run_id: str | None = None
    run_dir: Path | None = None
    last_error: str | None = None
    last_error_traceback: str | None = None
    log_lines: list[str] = None
    last_start_clicks: int = 0
    last_stop_clicks: int = 0

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
        self.last_error_traceback = None
        self.log_lines = []
        self.last_start_clicks = 0
        self.last_stop_clicks = 0


# Global singleton instance
_LIVE_SESSION = LiveSession()


def get_session() -> LiveSession:
    """Get the global live trading session."""
    return _LIVE_SESSION


def start_trading(args_list: list[str], run_id: str, run_dir: Path) -> None:
    """Start live trading in background thread."""
    session = get_session()

    # Guard: prevent multiple simultaneous starts
    if session.status == SessionStatus.RUNNING:
        raise RuntimeError("Trading session already running. Stop it first.")
    if session.status == SessionStatus.STOPPING:
        raise RuntimeError("Trading session is stopping. Wait for it to finish.")

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

            # Set to DONE when thread completes successfully
            # (don't override ERROR state)
            if session.status in (SessionStatus.RUNNING, SessionStatus.STOPPING):
                session.status = SessionStatus.DONE
        except Exception as e:
            session.status = SessionStatus.ERROR
            session.last_error = str(e)
            session.last_error_traceback = traceback.format_exc()
            logger.error(f"Live trading error: {e}", exc_info=True)

    session.thread = threading.Thread(target=_run_trading, daemon=True, name="LiveTradingThread")
    session.thread.start()


def stop_trading() -> None:
    """Stop live trading session."""
    session = get_session()

    if session.status not in (SessionStatus.RUNNING, SessionStatus.DONE):
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

    # Try to join thread with timeout (non-blocking check)
    if session.thread and session.thread.is_alive():
        # Thread will check kill switch in next loop iteration
        # Don't block here - let polling callback detect completion
        pass
