from __future__ import annotations


def test_kill_switch_triggers_on_drawdown() -> None:
    from autonomous_rl_trading_bot.risk.kill_switch import KillSwitch, KillSwitchConfig

    ks = KillSwitch(KillSwitchConfig(max_drawdown=0.1, min_equity=1.0))

    killed, reason = ks.check(equity=100.0, peak_equity=100.0)
    assert killed is False
    assert reason == ""

    killed, reason = ks.check(equity=85.0, peak_equity=100.0)
    assert killed is True
    assert reason == "max_drawdown"


def test_kill_switch_triggers_on_min_equity() -> None:
    from autonomous_rl_trading_bot.risk.kill_switch import KillSwitch, KillSwitchConfig

    ks = KillSwitch(KillSwitchConfig(max_drawdown=0.5, min_equity=50.0))

    killed, reason = ks.check(equity=49.0, peak_equity=100.0)
    assert killed is True
    assert reason == "min_equity"

