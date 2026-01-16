"""Windows desktop app wrapper for Dash dashboard using pywebview."""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests is not installed.")
    print("Install it with: pip install requests")
    sys.exit(1)


def _find_free_port(start_port: int = 8050, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port starting from {start_port}")


def _wait_for_server(url: str, timeout: float = 30.0, check_interval: float = 0.5) -> None:
    """Wait for server to respond with HTTP 200."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1.0)
            if response.status_code == 200:
                return
        except (requests.RequestException, ConnectionError):
            pass
        time.sleep(check_interval)
    raise RuntimeError(f"Server did not start within {timeout} seconds")


def _run_server(host: str, port: int, mode: str | None = None) -> None:
    """Run dashboard server in background thread."""
    # Import here to avoid issues if pywebview isn't installed
    from autonomous_rl_trading_bot.common.config import load_config
    from autonomous_rl_trading_bot.common.db import migrate
    from autonomous_rl_trading_bot.common.paths import ensure_artifact_tree
    from autonomous_rl_trading_bot.dashboard import create_app

    ensure_artifact_tree()
    loaded = load_config(mode=mode, base_name="base.yaml")
    cfg = loaded.config
    db_path = Path(migrate(cfg))
    app = create_app(db_path=db_path)

    # Run server (this blocks until shutdown)
    app.run(host=host, port=port, debug=False, use_reloader=False)


def main() -> int:
    """Main entrypoint for desktop app."""
    try:
        import webview
    except ImportError:
        print("ERROR: pywebview is not installed.")
        print("Install it with: pip install pywebview")
        return 1

    # Find available port
    port = _find_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    print(f"Starting dashboard server on {url}...")

    # Start server in background thread
    server_thread = threading.Thread(
        target=_run_server,
        args=(host, port),
        daemon=True,
        name="DashboardServerThread",
    )
    server_thread.start()

    # Wait for server to be ready
    try:
        print("Waiting for server to start...")
        _wait_for_server(url)
        print(f"Server ready at {url}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    # Open desktop window
    try:
        webview.create_window(
            title="Autonomous RL Trading Bot",
            url=url,
            width=1400,
            height=900,
            resizable=True,
            min_size=(800, 600),
        )
        webview.start(debug=False)
    except Exception as e:
        print(f"ERROR: Failed to create window: {e}")
        return 1

    print("Dashboard closed. Exiting...")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
