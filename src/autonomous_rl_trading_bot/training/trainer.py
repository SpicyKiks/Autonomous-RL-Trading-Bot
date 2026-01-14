# src/autonomous_rl_trading_bot/training/trainer.py
from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from autonomous_rl_trading_bot.common.fs import ensure_dir, write_json
from autonomous_rl_trading_bot.common.hashing import dataset_hash
from autonomous_rl_trading_bot.common.logging import get_logger
from autonomous_rl_trading_bot.evaluation.metrics import compute_metrics
from autonomous_rl_trading_bot.evaluation.reporting import generate_run_report
from autonomous_rl_trading_bot.repro.repro import build_repro_payload
from autonomous_rl_trading_bot.rl.dataset import Dataset
from autonomous_rl_trading_bot.rl.env_trading import TradingEnv, TradingEnvConfig

from stable_baselines3 import A2C, DQN, PPO

logger = get_logger(__name__)


def _asdict_dataclass(obj: Any) -> Dict[str, Any]:
    if not hasattr(obj, "__dataclass_fields__"):
        raise TypeError("Expected a dataclass instance")
    return {f.name: getattr(obj, f.name) for f in dc_fields(obj)}


def _json_serialize_paths(obj: Any) -> Any:
    """Recursively convert Path objects to strings for JSON serialization."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _json_serialize_paths(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_serialize_paths(item) for item in obj]
    else:
        return obj


# =========================
# Config
# =========================
@dataclass(frozen=True)
class TrainConfig:
    # dataset selection
    dataset_id: str

    # core training args
    mode: str  # "spot" | "futures"
    algo: str  # "ppo" | "a2c" | "dqn"
    timesteps: int
    seed: int

    # dataset split selection
    # Backwards compatible:
    # - may be a split name ("train"|"val"|"test") if dataset meta provides split ranges
    # - may be numeric weights (e.g., 8/2) to time-split the dataset
    # - may be None => auto (prefer meta splits if present; else time-split 80/20)
    train_split: Any = None
    eval_split: Any = None

    # env/hparams
    device: str = "cpu"
    lookback: int = 30
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    # capital / sizing
    initial_cash: float = 1000.0
    initial_equity: Optional[float] = None  # alias to initial_cash (wins if provided)
    position_fraction: float = 1.0  # [0,1]
    futures_leverage: float = 3.0  # only used for futures mode (if env supports)

    # reward
    reward_kind: str = "log_equity"  # log_equity|delta_equity (forwarded to env if supported)

    # outputs
    artifacts_dir: str = "artifacts"
    run_name: Optional[str] = None

    # optional explicit output locations (used by run_train.py plumbing)
    run_dir: Optional[Path] = None  # base runs dir OR fully qualified run output dir (see train_and_evaluate)
    models_dir: Optional[Path] = None

    # evaluation
    eval_episodes: int = 1

    # logging
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", str(self.mode).strip().lower())
        object.__setattr__(self, "algo", str(self.algo).strip().lower())
        object.__setattr__(self, "device", str(self.device).strip())

        if self.mode not in {"spot", "futures"}:
            raise ValueError("mode must be one of: spot, futures")

        if self.timesteps <= 0:
            raise ValueError("timesteps must be > 0")

        if self.lookback <= 0:
            raise ValueError("lookback must be > 0")

        if float(self.fee_bps) < 0:
            raise ValueError("fee_bps must be >= 0")

        if float(self.slippage_bps) < 0:
            raise ValueError("slippage_bps must be >= 0")

        if float(self.position_fraction) <= 0 or float(self.position_fraction) > 1.0:
            raise ValueError("position_fraction must be in (0, 1].")

        if float(self.futures_leverage) <= 0:
            raise ValueError("futures_leverage must be > 0.")

        # unify naming: if initial_equity is provided, it wins.
        if self.initial_equity is not None:
            object.__setattr__(self, "initial_cash", float(self.initial_equity))

        rk = str(self.reward_kind).strip().lower()
        object.__setattr__(self, "reward_kind", rk)


def _make_algo(algo: str):
    a = algo.lower().strip()
    if a == "ppo":
        return PPO
    if a == "a2c":
        return A2C
    if a == "dqn":
        return DQN
    raise ValueError(f"Unknown algo '{algo}'. Expected one of: ppo, a2c, dqn")


def _parse_split(v: Any) -> Tuple[str, Any]:
    """Return (kind, value) where kind in {'none','name','weight'}."""
    if v is None:
        return ("none", None)
    if isinstance(v, (int, float)):
        return ("weight", float(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "":
            return ("none", None)
        if s in {"train", "val", "valid", "validation", "test"}:
            # normalise common spellings
            if s in {"valid", "validation"}:
                s = "val"
            return ("name", s)
        # numeric string?
        try:
            f = float(s)
            return ("weight", f)
        except ValueError:
            return ("name", s)
    # unknown type: keep as name-ish (stringify) to avoid crashing on legacy inputs
    return ("name", str(v).strip().lower())


def _split_time_weighted(ds: Dataset, train_weight: float, eval_weight: float) -> Tuple[Dataset, Dataset]:
    if train_weight <= 0 or eval_weight <= 0:
        raise ValueError("train_split and eval_split weights must be > 0")
    total = float(train_weight + eval_weight)
    train_frac = float(train_weight) / total
    train_ds, eval_ds = ds.split_time(train_frac=train_frac)
    return train_ds, eval_ds


def _resolve_train_eval_datasets(cfg: TrainConfig, base_ds: Dataset) -> Tuple[Dataset, Dataset]:
    """Resolve train/eval datasets using either named splits (preferred) or a time split fallback."""
    train_kind, train_val = _parse_split(cfg.train_split)
    eval_kind, eval_val = _parse_split(cfg.eval_split)

    splits = {}
    try:
        splits = dict(base_ds.meta.get("splits") or {})
    except Exception:
        splits = {}

    # If any named split is requested OR the dataset provides split metadata, prefer split-based loading.
    if train_kind == "name" or eval_kind == "name" or splits:
        if not splits:
            # requested split names but dataset doesn't support it -> fall back to time split
            train_w = train_val if train_kind == "weight" else 8.0
            eval_w = eval_val if eval_kind == "weight" else 2.0
            return _split_time_weighted(base_ds, float(train_w), float(eval_w))

        train_name: Optional[str] = train_val if train_kind == "name" else None
        eval_name: Optional[str] = eval_val if eval_kind == "name" else None

        if train_name is None:
            train_name = "train" if "train" in splits else next(iter(splits.keys()))
        if eval_name is None:
            if "val" in splits:
                eval_name = "val"
            elif "test" in splits:
                eval_name = "test"
            else:
                # degenerate: only one split exists, reuse it for eval
                eval_name = train_name

        train_ds = Dataset.load(cfg.dataset_id, market_type=cfg.mode, split=train_name)
        eval_ds = Dataset.load(cfg.dataset_id, market_type=cfg.mode, split=eval_name)
        return train_ds, eval_ds

    # Default: time split (legacy)
    train_w = train_val if train_kind == "weight" else 8.0
    eval_w = eval_val if eval_kind == "weight" else 2.0
    return _split_time_weighted(base_ds, float(train_w), float(eval_w))


def _make_env_config(cfg: TrainConfig) -> TradingEnvConfig:
    """
    Build TradingEnvConfig but only pass keys it actually supports.
    This makes trainer resilient when env config changes.
    """
    env_cfg_kwargs: Dict[str, Any] = {
        "mode": cfg.mode,
        "lookback": cfg.lookback,
        "fee_bps": cfg.fee_bps,
        "slippage_bps": cfg.slippage_bps,
        "initial_cash": cfg.initial_cash,
        "seed": cfg.seed,
        "position_fraction": cfg.position_fraction,
        "futures_leverage": cfg.futures_leverage,
        "reward_kind": cfg.reward_kind,
    }

    allowed = {f.name for f in dc_fields(TradingEnvConfig)}
    filtered = {k: v for k, v in env_cfg_kwargs.items() if k in allowed}
    return TradingEnvConfig(**filtered)


def train_and_evaluate(
    *,
    cfg: TrainConfig,
    dataset: Optional[Dataset] = None,
    run_id: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    End-to-end training + evaluation run.

    Backwards/CLI-compatible plumbing:
    - run_train.py passes dataset/cfg/run_id/out_dir.
    - other callers may call train_and_evaluate(cfg=...).
    """
    # --- dataset ---
    ds = dataset if dataset is not None else Dataset.load(cfg.dataset_id, market_type=cfg.mode, split=None)
    ds_hash = dataset_hash(ds)

    # --- naming / folders ---
    if out_dir is not None:
        run_dir = Path(out_dir)
        ensure_dir(run_dir)
        run_name = cfg.run_name or (run_id if run_id is not None else run_dir.name)
    else:
        artifacts_root = Path(cfg.artifacts_dir)
        run_name = cfg.run_name or (
            run_id if run_id is not None else f"{cfg.mode}_{cfg.algo}_seed{cfg.seed}_{ds_hash[:8]}"
        )
        run_dir = artifacts_root / run_name
        ensure_dir(run_dir)

    # --- split ---
    train_ds, eval_ds = _resolve_train_eval_datasets(cfg, ds)

    # --- env config ---
    env_cfg = _make_env_config(cfg)

    # --- build envs ---
    train_env = TradingEnv(train_ds.data, env_cfg)
    eval_env = TradingEnv(eval_ds.data, env_cfg)

    # --- model ---
    AlgoCls = _make_algo(cfg.algo)
    model = AlgoCls(
        "MlpPolicy",
        train_env,
        verbose=0,
        device=cfg.device,
        seed=cfg.seed,
    )

    # --- fit ---
    model.learn(total_timesteps=int(cfg.timesteps))

    # --- save model ---
    policy_path = run_dir / "policy.zip"
    model.save(str(policy_path))
    logger.info(f"Saved trained model to: {policy_path}")

    # --- eval ---
    all_episode_infos: list[Dict[str, Any]] = []
    for _ in range(int(cfg.eval_episodes)):
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            if isinstance(info, dict):
                all_episode_infos.append(info)

    # Extract data for compute_metrics
    if not all_episode_infos:
        raise ValueError("No episode info collected during evaluation")
    
    # Get timestamps from dataset (use indices from info['t'] to map to open_time_ms)
    open_time_ms_array = eval_ds.data.get("open_time_ms")
    if open_time_ms_array is None:
        # Fallback: generate timestamps from step indices
        # Assume 1-minute intervals if not available
        interval_ms_fallback = 60_000  # 1 minute default
        start_time = 0
        open_time_ms_list = [start_time + i * interval_ms_fallback for i in range(len(all_episode_infos))]
    else:
        # Convert numpy array to list and map info['t'] indices to actual timestamps
        if isinstance(open_time_ms_array, np.ndarray):
            open_time_ms_array = open_time_ms_array.tolist()
        open_time_ms_list = []
        for info in all_episode_infos:
            t_idx = int(info.get("t", 0))
            if t_idx < len(open_time_ms_array):
                open_time_ms_list.append(int(open_time_ms_array[t_idx]))
            else:
                open_time_ms_list.append(int(open_time_ms_array[-1]) if open_time_ms_array else 0)
    
    equity_list = [float(info.get("equity", 0.0)) for info in all_episode_infos]
    
    # Calculate drawdown
    peak_equity = equity_list[0] if equity_list else 1.0
    drawdown_list: list[float] = []
    for eq in equity_list:
        peak_equity = max(peak_equity, eq)
        dd = max(0.0, 1.0 - (eq / peak_equity)) if peak_equity > 0 else 0.0
        drawdown_list.append(dd)
    
    # Get fee_total and slippage_total from last info
    last_info = all_episode_infos[-1] if all_episode_infos else {}
    fee_total = float(last_info.get("fee_total", 0.0))
    slippage_total = float(last_info.get("slippage_total", 0.0))
    
    # Get interval_ms from dataset meta
    interval_ms = int(eval_ds.meta.get("interval_ms", 60_000))  # Default 1 minute
    
    # Count trades (approximate: count non-zero position changes)
    # For now, set to 0 since we don't track explicit trades in TradingEnv
    trade_count = 0
    
    metrics = compute_metrics(
        open_time_ms=open_time_ms_list,
        equity=equity_list,
        drawdown=drawdown_list,
        trade_count=trade_count,
        fee_total=fee_total,
        slippage_total=slippage_total,
        interval_ms=interval_ms,
        trades=None,
        exposure=None,
        seed=cfg.seed,
    )

    # --- repro payload ---
    repro = build_repro_payload(
        config=_asdict_dataclass(cfg),
        extra={
            "dataset_id": cfg.dataset_id,
            "dataset_hash": ds_hash,
            "metrics": metrics.to_dict() if hasattr(metrics, "to_dict") else metrics,
        },
    )

    # --- write artifacts ---
    cfg_dict = _asdict_dataclass(cfg)
    write_json(run_dir / "config.json", _json_serialize_paths(cfg_dict))
    write_json(run_dir / "dataset.json", {"dataset_id": cfg.dataset_id, "dataset_hash": ds_hash})
    write_json(run_dir / "metrics.json", metrics.to_dict() if hasattr(metrics, "to_dict") else metrics)
    write_json(run_dir / "repro.json", _json_serialize_paths(repro))

    # Convert episode infos to equity_rows format for reporting
    equity_rows = [
        {
            "open_time_ms": ts,
            "equity": eq,
            "drawdown": dd,
        }
        for ts, eq, dd in zip(open_time_ms_list, equity_list, drawdown_list)
    ]
    
    # Empty trades list (TradingEnv doesn't track explicit trades)
    trades_rows: list[Dict[str, Any]] = []
    
    report_artifacts = generate_run_report(
        out_dir=run_dir,
        title=f"Training Run: {run_name}",
        equity_rows=equity_rows,
        trades_rows=trades_rows,
        metrics=metrics.to_dict() if hasattr(metrics, "to_dict") else metrics,
        repro=_json_serialize_paths(repro),
    )

    payload: Dict[str, Any] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "dataset_id": cfg.dataset_id,
        "dataset_hash": ds_hash,
        "metrics": metrics.to_dict() if hasattr(metrics, "to_dict") else metrics,
        "report_path": str(report_artifacts.get("report_html", "")) if report_artifacts else None,
    }
    return payload
