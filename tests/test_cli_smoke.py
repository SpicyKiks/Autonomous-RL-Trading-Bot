from __future__ import annotations

import pytest

from autonomous_rl_trading_bot import cli


def test_cli_version():
    """Test --version flag."""
    result = cli.main(["--version"])
    assert result == 0


def test_cli_help(capsys):
    """Test --help flag and verify backtest appears."""
    result = cli.main(["--help"])
    assert result == 0
    captured = capsys.readouterr()
    assert "backtest" in captured.out.lower()


def test_cli_parse_args_backtest():
    """Test argument parsing for backtest command."""
    # --help causes argparse to call sys.exit(), so args will be None
    args, remainder = cli._parse_args(["backtest", "--help"])
    # When --help is present, argparse calls sys.exit() so args is None
    assert args is None
    assert remainder == []


def test_cli_parse_args_backtest_with_args():
    """Test that backtest command forwards remaining args."""
    args, remainder = cli._parse_args(["backtest", "--mode", "spot"])
    assert args is not None
    assert args.command == "backtest"
    assert "--mode" in remainder
    assert "spot" in remainder


def test_cli_parse_args_backtest_with_double_dash():
    """Test forwarding behavior with -- separator."""
    args, remainder = cli._parse_args(["backtest", "--", "--help"])
    assert args is not None
    assert args.command == "backtest"
    assert "--help" in remainder


def test_cli_dataset_build_help():
    """Test dataset build subcommand help."""
    # argparse calls sys.exit(0) for --help, which pytest catches
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["dataset", "build", "--help"])
    assert exc_info.value.code == 0


def test_cli_no_command_shows_help(capsys):
    """Test that no command shows main help."""
    result = cli.main([])
    assert result == 0
    captured = capsys.readouterr()
    assert "COMMAND" in captured.out or "usage" in captured.out.lower()
