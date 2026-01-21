"""Process manager for live trading subprocess control."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autonomous_rl_trading_bot.common.paths import artifacts_dir, ensure_dir


class LiveProcessManager:
    """Manages live trading subprocess lifecycle."""

    def __init__(self, logs_dir: Path | None = None):
        """
        Initialize process manager.
        
        Args:
            logs_dir: Directory for logs and state files (default: artifacts/logs/live)
        """
        if logs_dir is None:
            logs_dir = artifacts_dir() / "logs" / "live"
        self.logs_dir = Path(logs_dir)
        ensure_dir(self.logs_dir)
        
        self._process: subprocess.Popen[str] | None = None
        self._run_id: str | None = None
        self._started_at: float | None = None
        self._log_path: Path | None = None
        self._state_path: Path | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_heartbeat = threading.Event()

    def start_live(self, args: dict[str, Any]) -> str:
        """
        Start live trading subprocess.
        
        Args:
            args: Dictionary of arguments to pass to live runner
                 (mode, symbol, interval, demo, policy, strategy, etc.)
        
        Returns:
            run_id: Unique run identifier
        
        Raises:
            RuntimeError: If already running
        """
        if self.is_running():
            raise RuntimeError("Live trading is already running. Stop it first.")
        
        # Generate run_id
        from autonomous_rl_trading_bot.common.config import load_config
        from autonomous_rl_trading_bot.common.run_ids import generate_run_id
        
        mode = args.get("mode", "spot")
        symbol = args.get("symbol", "BTCUSDT")
        interval = args.get("interval", "1m")
        
        # Load config to get hash
        loaded = load_config(mode=mode)
        cfg_hash = loaded.config_hash
        
        # Generate run_id
        run_id = generate_run_id(
            kind="live",
            mode=mode,
            symbol=symbol,
            interval=interval,
            cfg_hash=cfg_hash,
        )
        
        # Set up paths
        self._run_id = run_id
        self._log_path = self.logs_dir / f"{run_id}.log"
        self._state_path = self.logs_dir / f"{run_id}_state.json"
        self._started_at = time.time()
        
        # Build command: python -m autonomous_rl_trading_bot live --args...
        # Use the canonical command module
        cmd = [
            sys.executable,
            "-m",
            "autonomous_rl_trading_bot",
            "live",
        ]
        
        # Add arguments
        if mode:
            cmd.extend(["--mode", str(mode)])
        if symbol:
            cmd.extend(["--symbol", str(symbol)])
        if interval:
            cmd.extend(["--interval", str(interval)])
        if args.get("demo", True):  # Default to demo=True
            cmd.append("--demo")
        if args.get("policy"):
            cmd.extend(["--policy", str(args["policy"])])
        if args.get("strategy"):
            cmd.extend(["--strategy", str(args["strategy"])])
        if args.get("sb3_algo"):
            cmd.extend(["--sb3_algo", str(args["sb3_algo"])])
        if args.get("sb3_model_path"):
            cmd.extend(["--sb3_model_path", str(args["sb3_model_path"])])
        if args.get("max_steps"):
            cmd.extend(["--max_steps", str(int(args["max_steps"]))])
        if args.get("max_minutes"):
            cmd.extend(["--max_minutes", str(float(args["max_minutes"]))])
        
        # Open log file
        log_file = self._log_path.open("w", encoding="utf-8")
        
        # Start subprocess
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                cwd=Path.cwd(),
            )
        except Exception as e:
            log_file.close()
            raise RuntimeError(f"Failed to start live trading process: {e}") from e
        
        # Write initial state
        self._write_state({
            "status": "running",
            "run_id": run_id,
            "pid": self._process.pid,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "equity": None,
            "position": None,
            "pnl": None,
            "last_action": None,
        })
        
        # Start heartbeat thread to periodically update state from run_dir
        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="HeartbeatThread",
        )
        self._heartbeat_thread.start()
        
        return run_id

    def stop_live(self) -> bool:
        """
        Stop live trading subprocess.
        
        Returns:
            True if stopped successfully, False if not running
        """
        if not self.is_running():
            return False
        
        if self._process is None:
            return False
        
        # Update state to stopping
        self._write_state({
            "status": "stopping",
            "run_id": self._run_id,
            "pid": self._process.pid,
            "started_at": datetime.fromtimestamp(self._started_at, tz=timezone.utc).isoformat() if self._started_at else None,
        })
        
        # Windows-safe termination
        try:
            if platform.system() == "Windows":
                # Use taskkill for Windows (terminates process tree)
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(self._process.pid)],
                    check=False,
                    capture_output=True,
                )
            else:
                # Unix: send SIGTERM, then SIGKILL if needed
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
        except Exception as e:
            # Log error but continue
            if self._log_path:
                try:
                    with self._log_path.open("a", encoding="utf-8") as f:
                        f.write(f"\n[ERROR] Failed to stop process: {e}\n")
                except Exception:
                    pass
        
        # Wait for process to terminate
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        
        # Update state to stopped
        self._write_state({
            "status": "stopped",
            "run_id": self._run_id,
            "pid": self._process.pid,
            "started_at": datetime.fromtimestamp(self._started_at, tz=timezone.utc).isoformat() if self._started_at else None,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
        })
        
        # Stop heartbeat thread
        self._stop_heartbeat.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2)
        
        # Clean up
        self._process = None
        self._run_id = None
        self._started_at = None
        self._heartbeat_thread = None
        
        return True

    def status(self) -> dict[str, Any]:
        """
        Get current status.
        
        Returns:
            Dictionary with: running, run_id, pid, started_at, log_path, state_path
        """
        running = self.is_running()
        
        result: dict[str, Any] = {
            "running": running,
            "run_id": self._run_id,
            "pid": self._process.pid if self._process else None,
            "started_at": datetime.fromtimestamp(self._started_at, tz=timezone.utc).isoformat() if self._started_at else None,
            "log_path": str(self._log_path) if self._log_path else None,
            "state_path": str(self._state_path) if self._state_path else None,
        }
        
        # Load state file if exists
        if self._state_path and self._state_path.exists():
            try:
                state_data = json.loads(self._state_path.read_text(encoding="utf-8"))
                result.update(state_data)
            except Exception:
                pass
        
        return result

    def tail_logs(self, n: int = 200) -> list[str]:
        """
        Get last N lines from log file.
        
        Args:
            n: Number of lines to return
        
        Returns:
            List of log lines (most recent last)
        """
        if not self._log_path or not self._log_path.exists():
            return []
        
        try:
            with self._log_path.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                return lines[-n:] if len(lines) > n else lines
        except Exception:
            return []

    def is_running(self) -> bool:
        """Check if process is running."""
        if self._process is None:
            return False
        return self._process.poll() is None  # None means still running

    def _write_state(self, state: dict[str, Any]) -> None:
        """Write state JSON file."""
        if not self._state_path:
            return
        
        try:
            state["timestamp"] = datetime.now(timezone.utc).isoformat()
            self._state_path.write_text(
                json.dumps(state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass  # Silently fail if can't write state

    def _heartbeat_loop(self) -> None:
        """Background thread to periodically update state from run_dir."""
        while not self._stop_heartbeat.is_set():
            try:
                if self._run_id and self._process and self._process.poll() is None:
                    # Process is running, try to read state from run_dir
                    from autonomous_rl_trading_bot.common.paths import artifacts_dir
                    run_dir = artifacts_dir() / "runs" / self._run_id
                    
                    if run_dir.exists():
                        # Try to read latest equity from equity.csv
                        equity_csv = run_dir / "equity.csv"
                        equity = None
                        position = None
                        last_action = None
                        
                        if equity_csv.exists():
                            try:
                                import pandas as pd
                                df = pd.read_csv(equity_csv)
                                if not df.empty:
                                    equity = float(df["equity"].iloc[-1])
                                    # Try to infer position from exposure or other columns
                                    if "position_qty" in df.columns:
                                        position = float(df["position_qty"].iloc[-1])
                            except Exception:
                                pass
                        
                        # Try to read metrics.json
                        metrics_json = run_dir / "metrics.json"
                        pnl = None
                        if metrics_json.exists():
                            try:
                                metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
                                # Calculate PnL from metrics if available
                                if "final_equity" in metrics and equity:
                                    initial_equity = metrics.get("initial_equity", equity)
                                    pnl = equity - initial_equity
                            except Exception:
                                pass
                        
                        # Try to read last trade from trades.csv
                        trades_csv = run_dir / "trades.csv"
                        if trades_csv.exists():
                            try:
                                import pandas as pd
                                df = pd.read_csv(trades_csv)
                                if not df.empty:
                                    last_side = df["side"].iloc[-1]
                                    last_action = f"{last_side.upper()}"
                            except Exception:
                                pass
                        
                        # Update state file
                        current_state = {
                            "status": "running",
                            "run_id": self._run_id,
                            "pid": self._process.pid,
                            "started_at": datetime.fromtimestamp(self._started_at, tz=timezone.utc).isoformat() if self._started_at else None,
                            "equity": equity,
                            "position": position,
                            "pnl": pnl,
                            "last_action": last_action,
                        }
                        self._write_state(current_state)
                
                # Sleep for 2 seconds before next update
                if self._stop_heartbeat.wait(timeout=2):
                    break
            except Exception:
                # Continue heartbeat loop even if update fails
                if self._stop_heartbeat.wait(timeout=2):
                    break
