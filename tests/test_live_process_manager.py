"""Tests for live process manager (no network required)."""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

from autonomous_rl_trading_bot.services.live_process_manager import LiveProcessManager


def test_process_manager_start_stop():
    """Test starting and stopping a dummy process."""
    with TemporaryDirectory() as tmpdir:
        manager = LiveProcessManager(logs_dir=Path(tmpdir))
        
        # Initially not running
        assert not manager.is_running()
        status = manager.status()
        assert status["running"] is False
        assert status["run_id"] is None
        
        # Test with a simple Python command that prints lines
        import sys
        test_script = """import time
import sys
for i in range(10):
    print(f"Line {i}", flush=True)
    time.sleep(0.1)
"""
        test_script_path = Path(tmpdir) / "test_script.py"
        test_script_path.write_text(test_script, encoding="utf-8")
        
        # Manually start a process to test process management
        import subprocess
        log_file = (Path(tmpdir) / "test.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(test_script_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=tmpdir,
        )
        
        # Test that we can track a process
        manager._process = process
        manager._run_id = "test_run_123"
        manager._log_path = Path(tmpdir) / "test.log"
        manager._state_path = Path(tmpdir) / "test_state.json"
        manager._started_at = time.time()
        
        assert manager.is_running()
        
        # Wait a bit for process to produce output
        time.sleep(0.5)
        
        # Verify logs are being written
        if manager._log_path.exists():
            log_content = manager._log_path.read_text(encoding="utf-8")
            assert "Line" in log_content
        
        # Stop the process
        success = manager.stop_live()
        assert success
        
        # Process should be stopped
        assert not manager.is_running()
        
        # Status should reflect stopped state
        status = manager.status()
        assert status["running"] is False
        
        # Close log file
        log_file.close()


def test_process_manager_logs():
    """Test log tailing functionality."""
    with TemporaryDirectory() as tmpdir:
        manager = LiveProcessManager(logs_dir=Path(tmpdir))
        
        # Create a test log file
        log_path = Path(tmpdir) / "test_run.log"
        log_lines = [f"Log line {i}\n" for i in range(100)]
        log_path.write_text("".join(log_lines), encoding="utf-8")
        
        manager._log_path = log_path
        
        # Get last 20 lines
        tail = manager.tail_logs(n=20)
        assert len(tail) == 20
        assert "Log line 80" in tail[0]
        assert "Log line 99" in tail[-1]
        
        # Get more lines than exist
        tail_all = manager.tail_logs(n=200)
        assert len(tail_all) == 100


def test_process_manager_state():
    """Test state file writing."""
    with TemporaryDirectory() as tmpdir:
        manager = LiveProcessManager(logs_dir=Path(tmpdir))
        
        state_path = Path(tmpdir) / "test_state.json"
        manager._state_path = state_path
        
        # Write state
        test_state = {
            "status": "running",
            "run_id": "test_123",
            "equity": 1000.0,
            "position": 0.5,
            "pnl": 10.5,
            "last_action": "BUY",
        }
        manager._write_state(test_state)
        
        # Verify state file exists and contains correct data
        assert state_path.exists()
        import json
        loaded_state = json.loads(state_path.read_text(encoding="utf-8"))
        assert loaded_state["status"] == "running"
        assert loaded_state["run_id"] == "test_123"
        assert loaded_state["equity"] == 1000.0
        assert "timestamp" in loaded_state


def test_process_manager_status():
    """Test status reporting."""
    with TemporaryDirectory() as tmpdir:
        manager = LiveProcessManager(logs_dir=Path(tmpdir))
        
        # Initial status
        status = manager.status()
        assert status["running"] is False
        assert status["run_id"] is None
        assert status["pid"] is None
        
        # Set up a mock process state
        manager._run_id = "test_run"
        manager._log_path = Path(tmpdir) / "test.log"
        manager._state_path = Path(tmpdir) / "test_state.json"
        manager._started_at = time.time()
        
        # Create state file
        import json
        state_data = {
            "status": "running",
            "equity": 1000.0,
            "position": 0.5,
        }
        manager._state_path.write_text(json.dumps(state_data), encoding="utf-8")
        
        # Get status (should include state data)
        status = manager.status()
        assert status["run_id"] == "test_run"
        assert status["log_path"] is not None
        assert status["state_path"] is not None
