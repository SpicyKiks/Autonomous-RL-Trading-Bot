from __future__ import annotations

import pytest

from autonomous_rl_trading_bot import cli


def test_cli_version():
    """Test --version flag."""
    # argparse calls sys.exit(0) for --version, which pytest catches
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0


def test_cli_help(capsys):
    """Test --help flag and verify backtest appears."""
    # argparse calls sys.exit(0) for --help, which pytest catches
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "backtest" in captured.out.lower()


# Note: _parse_args() is not part of the public API, so these tests are removed
# The CLI behavior is tested via the main() function which is the public interface


def test_cli_dataset_build_help():
    """Test dataset build subcommand help."""
    # argparse calls sys.exit(0) for --help, which pytest catches
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["dataset", "build", "--help"])
    assert exc_info.value.code == 0


def test_cli_no_command_shows_help(capsys):
    """Test that no command shows main help."""
    # argparse calls sys.exit(2) when required argument is missing
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "COMMAND" in captured.out or "usage" in captured.out.lower() or "required" in captured.err.lower()
