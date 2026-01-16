from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from autonomous_rl_trading_bot.common.hashing import dataset_hash as compute_dataset_hash
from autonomous_rl_trading_bot.common.logging import get_logger
from autonomous_rl_trading_bot.common.paths import artifacts_dir
from autonomous_rl_trading_bot.evaluation.backtester import (
    BacktestConfig,
    run_futures_backtest,
    run_spot_backtest,
)
from autonomous_rl_trading_bot.evaluation.baselines import Strategy, make_strategy
from autonomous_rl_trading_bot.evaluation.reporting import (
    build_repro_payload,
    generate_run_report,
)

log = get_logger("arbt")


# ─────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────
def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_backtest_run_id(mode: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"run_backtest_{mode}_{ts}_{day}"


def _resolve_run_dir(run_id: str) -> Path:
    return artifacts_dir() / "runs" / run_id


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_error(path: Path, exc: BaseException) -> None:
    path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Dataset discovery (robust, no assumptions)
# ─────────────────────────────────────────────────────────────
def _dataset_root_for_id(dataset_id: str) -> Path:
    # Try artifacts/datasets first (standard location), then artifacts/runs (legacy)
    root = artifacts_dir() / "datasets" / dataset_id
    if root.exists():
        return root
    root = artifacts_dir() / "runs" / dataset_id
    if root.exists():
        return root
    raise FileNotFoundError(f"Dataset root not found for dataset_id={dataset_id}. Checked: {artifacts_dir() / 'datasets' / dataset_id} and {artifacts_dir() / 'runs' / dataset_id}")


def _find_dataset_files(root: Path) -> Tuple[Path, Path]:
    """
    Search recursively under <root> for meta.json + dataset.npz.
    Returns (meta_path, npz_path).

    This avoids breaking if your dataset writer stores files in subfolders.
    """
    meta = root / "meta.json"
    npz = root / "dataset.npz"
    if meta.exists() and npz.exists():
        return meta, npz

    metas = list(root.rglob("meta.json"))
    npzs = list(root.rglob("dataset.npz"))

    if not metas or not npzs:
        raise FileNotFoundError(
            "Could not find required dataset files under dataset_id directory.\n"
            f"Looked under: {root}\n"
            f"Found meta.json: {len(metas)} | Found dataset.npz: {len(npzs)}"
        )

    # Prefer pairs that live in the same folder
    npz_by_parent = {p.parent: p for p in npzs}
    for m in metas:
        if m.parent in npz_by_parent:
            return m, npz_by_parent[m.parent]

    # Otherwise pick the closest-ish pair
    best = None
    best_score = None
    for m in metas:
        for z in npzs:
            score = abs(len(str(m.parent)) - len(str(z.parent)))
            if best_score is None or score < best_score:
                best_score = score
                best = (m, z)

    assert best is not None
    return best


def _load_dataset_from_id(dataset_id: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    root = _dataset_root_for_id(dataset_id)
    meta_path, npz_path = _find_dataset_files(root)

    dataset_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays_npz = np.load(npz_path, allow_pickle=False)
    arrays: Dict[str, np.ndarray] = {k: arrays_npz[k] for k in arrays_npz.files}

    dataset_meta = dict(dataset_meta)
    dataset_meta["_dataset_root"] = str(root)
    dataset_meta["_meta_path"] = str(meta_path)
    dataset_meta["_npz_path"] = str(npz_path)
    dataset_meta["dataset_id"] = dataset_id
    return dataset_meta, arrays


# ─────────────────────────────────────────────────────────────
# Config override handling (keeps CLI compatibility)
# ─────────────────────────────────────────────────────────────
def _cfg_from_meta(dataset_meta: Mapping[str, Any]) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=float(dataset_meta.get("initial_cash", 10_000.0)),
        order_size_quote=float(dataset_meta.get("order_size_quote", 0.0)),
        taker_fee_rate=float(dataset_meta.get("taker_fee_rate", 0.0004)),
        slippage_bps=float(dataset_meta.get("slippage_bps", 0.0)),
        leverage=float(dataset_meta.get("leverage", 1.0)),
        maintenance_margin_rate=float(dataset_meta.get("maintenance_margin_rate", 0.005)),
        allow_short=bool(dataset_meta.get("allow_short", True)),
        stop_on_liquidation=bool(dataset_meta.get("stop_on_liquidation", True)),
    )


def _apply_cfg_override(base: BacktestConfig, override: Any) -> BacktestConfig:
    """
    override can be:
      - None
      - BacktestConfig
      - dict-like with BacktestConfig field names
    Anything else is ignored (but logged).
    """
    if override is None:
        return base

    if isinstance(override, BacktestConfig):
        return override

    if isinstance(override, Mapping):
        allowed = set(asdict(base).keys())
        patch = {k: override[k] for k in override.keys() if k in allowed}
        if not patch:
            return base
        # dataclasses.replace keeps types/fields safe
        return replace(base, **patch)

    log.warning(f"Ignoring unsupported cfg override type: {type(override)}")
    return base


# ─────────────────────────────────────────────────────────────
# SB3 policy loader + wrapper Strategy
# ─────────────────────────────────────────────────────────────
def _try_load_sb3_model(policy_path: Path):
    from stable_baselines3 import A2C, DQN, PPO, SAC, TD3  # type: ignore

    errors: List[str] = []
    for cls in (PPO, A2C, DQN, SAC, TD3):
        try:
            # Use CPU for MLP policies (faster and avoids GPU warning)
            return cls.load(str(policy_path), device="cpu")
        except Exception as e:
            errors.append(f"{cls.__name__}: {e}")
    raise ValueError(
        "Failed to load SB3 policy.zip with PPO/A2C/DQN/SAC/TD3.\n" + "\n".join(errors)
    )


def _build_obs_matrix(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if "obs" in arrays:
        obs = np.asarray(arrays["obs"])
    elif "features" in arrays:
        obs = np.asarray(arrays["features"])
    elif "x" in arrays:
        obs = np.asarray(arrays["x"])
    else:
        if "close" not in arrays:
            raise KeyError(
                "Cannot build observations: none of ['obs','features','x'] exist and 'close' missing too."
            )
        close = np.asarray(arrays["close"], dtype=np.float32)
        obs = close.reshape(-1, 1)

    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    return obs.astype(np.float32, copy=False)


class SB3PolicyStrategy(Strategy):
    def __init__(self, model: Any, obs_matrix: np.ndarray):
        self._model = model
        self._obs = obs_matrix
        
        # Get lookback from model's observation space
        # TradingEnv uses (lookback, n_features) shape
        obs_space = model.observation_space
        if hasattr(obs_space, 'shape') and len(obs_space.shape) == 2:
            # Shape is (lookback, n_features)
            self._lookback = int(obs_space.shape[0])
            self._n_features = int(obs_space.shape[1])
        else:
            # Fallback: try to infer from observation space or use default
            # If it's a flattened Box, we need to infer lookback from data
            if hasattr(obs_space, 'shape') and len(obs_space.shape) == 1:
                # Flattened: might be (lookback * n_features + account_dim,)
                # Try to infer from obs_matrix shape
                if obs_matrix.ndim == 2:
                    self._lookback = 30  # default
                    self._n_features = int(obs_matrix.shape[1]) if obs_matrix.shape[1] > 0 else 1
                else:
                    self._lookback = 30
                    self._n_features = 1
            else:
                # Default fallback
                self._lookback = 30
                self._n_features = int(obs_matrix.shape[1]) if obs_matrix.ndim == 2 and obs_matrix.shape[1] > 0 else 1

    def act(self, t: int, price: float) -> int:
        if t < 0:
            t = 0
        if t >= len(self._obs):
            t = len(self._obs) - 1

        # Build lookback window similar to TradingEnv._get_obs()
        lb = self._lookback
        t0 = max(t - lb + 1, 0)
        t1 = t + 1
        
        # Extract window from observation matrix
        if self._obs.ndim == 2:
            # obs_matrix is (n_timesteps, n_features)
            window = self._obs[t0:t1]
        else:
            # obs_matrix is 1D, reshape to (n_timesteps, 1)
            window = self._obs[t0:t1].reshape(-1, 1)
        
        # Pad at the top if needed (when t < lookback-1)
        if window.shape[0] < lb:
            pad_shape = (lb - window.shape[0], window.shape[1] if window.ndim == 2 else 1)
            pad = np.zeros(pad_shape, dtype=np.float32)
            if window.ndim == 2:
                window = np.vstack([pad, window])
            else:
                window = np.concatenate([pad.reshape(-1), window])
                window = window.reshape(-1, 1)
        
        # Ensure correct shape: (lookback, n_features)
        if window.ndim == 1:
            window = window.reshape(-1, 1)
        if window.shape[0] != lb:
            # Trim or pad to exact lookback size
            if window.shape[0] > lb:
                window = window[-lb:]
            else:
                pad = np.zeros((lb - window.shape[0], window.shape[1]), dtype=np.float32)
                window = np.vstack([pad, window])
        
        obs_in = window.astype(np.float32, copy=False)
        
        # Validate shape matches observation space
        expected_shape = (self._lookback, self._n_features)
        if obs_in.shape != expected_shape:
            raise ValueError(
                f"Observation shape mismatch: got {obs_in.shape}, expected {expected_shape}. "
                f"Model observation_space: {self._model.observation_space}"
            )
        
        action, _ = self._model.predict(obs_in, deterministic=True)

        a = action
        if isinstance(a, np.ndarray):
            a = a.reshape(-1)[0] if a.size else 0

        # Discrete actions (expected)
        try:
            ai = int(a)
            if 0 <= ai <= 3:
                return ai
        except Exception:
            pass

        # Continuous fallback
        try:
            af = float(a)
        except Exception:
            return 0
        if af > 0.33:
            return 1
        if af < -0.33:
            return 2
        return 0


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def run_backtest(
    *,
    mode: str,
    dataset_id: str,
    run_id: Optional[str] = None,
    policy: Optional[str] = None,
    strategies_cfg: Optional[Mapping[str, Any]] = None,
    cfg: Optional[Union[BacktestConfig, Mapping[str, Any]]] = None,
    **_ignored: Any,  # swallow extra kwargs from CLI to prevent future crashes
) -> Dict[str, Any]:
    mode = str(mode).lower().strip()
    if mode not in ("spot", "futures"):
        raise ValueError(f"mode must be 'spot' or 'futures', got: {mode}")

    if not dataset_id:
        raise ValueError("dataset_id is required")

    resolved_run_id = run_id or _new_backtest_run_id(mode)
    out_dir = _resolve_run_dir(resolved_run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    started_utc = _utc_iso()

    try:
        dataset_meta, arrays = _load_dataset_from_id(dataset_id)

        # cfg from meta, then override from CLI if provided
        cfg_obj = _cfg_from_meta(dataset_meta)
        cfg_obj = _apply_cfg_override(cfg_obj, cfg)

        # Strategy/policy selection
        policy_arg = (policy or "").strip()
        if policy_arg == "" or policy_arg.lower() == "baseline":
            strategy_name = "buy_and_hold"
        else:
            strategy_name = policy_arg

        params: Dict[str, Any] = {}
        strategy_used: str

        p = Path(strategy_name)
        if p.suffix.lower() == ".zip" and p.exists():
            log.info(f"Loading SB3 policy: {p}")
            model = _try_load_sb3_model(p)
            obs = _build_obs_matrix(arrays)
            strategy: Strategy = SB3PolicyStrategy(model=model, obs_matrix=obs)
            strategy_used = str(p)
            params = {"policy_type": "sb3", "policy_path": str(p), "obs_shape": list(obs.shape)}
        else:
            strategy = make_strategy(strategy_name, **(dict(strategies_cfg or {})))
            strategy_used = strategy_name
            params = {"strategy": strategy_name, **(dict(strategies_cfg or {}))}

        run_input = {
            "run_id": resolved_run_id,
            "mode": mode,
            "dataset_id": dataset_id,
            "policy": policy,
            "strategy_used": strategy_used,
            "params": params,
            "cfg": asdict(cfg_obj),
            "started_utc": started_utc,
            "dataset_debug": {
                "root": dataset_meta.get("_dataset_root"),
                "meta_path": dataset_meta.get("_meta_path"),
                "npz_path": dataset_meta.get("_npz_path"),
                "npz_keys": list(arrays.keys()),
            },
        }
        _write_json(out_dir / "run_input.json", run_input)

        if mode == "spot":
            equity_rows, trade_rows, metrics, extra = run_spot_backtest(
                dataset_meta=dataset_meta,
                arrays=arrays,
                strategy=strategy,
                cfg=cfg_obj,
            )
        else:
            equity_rows, trade_rows, metrics, extra = run_futures_backtest(
                dataset_meta=dataset_meta,
                arrays=arrays,
                strategy=strategy,
                cfg=cfg_obj,
            )

        finished_utc = _utc_iso()

        # Compute dataset hash for reproducibility
        ds_hash = compute_dataset_hash(dataset_meta, length=10)
        
        # Build reproducibility payload
        seed = int(dataset_meta.get("seed", 1337))
        repro = build_repro_payload(
            seed=seed,
            dataset_id=dataset_id,
            dataset_hash=ds_hash,
            kind="backtest",
            run_id=resolved_run_id,
            mode=mode,
            config_hash=None,
            extra={
                "policy": policy,
                "strategy_used": strategy_used,
                "params": params,
            },
        )

        run_json = {
            **run_input,
            "finished_utc": finished_utc,
            "status": "ok",
            "metrics": metrics.to_dict(),
            "extra": extra,
            "artifacts_dir": str(out_dir),
            "equity_rows": equity_rows,
            "trade_rows": trade_rows,
        }
        _write_json(out_dir / "run.json", run_json)

        # Generate reporting artifacts (same as training)
        title = f"Backtest: {strategy_used} on {dataset_id}"
        generate_run_report(
            out_dir=out_dir,
            title=title,
            equity_rows=equity_rows,
            trades_rows=trade_rows,
            metrics=metrics.to_dict(),
            repro=repro,
        )

        return {
            "run_id": resolved_run_id,
            "mode": mode,
            "dataset_id": dataset_id,
            "policy": policy,
            "strategy_used": strategy_used,
            "status": "ok",
            "artifacts_dir": str(out_dir),
        }

    except Exception as e:
        finished_utc = _utc_iso()
        _write_error(out_dir / "error.txt", e)
        try:
            _write_json(
                out_dir / "run.json",
                {
                    "run_id": resolved_run_id,
                    "mode": mode,
                    "dataset_id": dataset_id,
                    "policy": policy,
                    "started_utc": started_utc,
                    "finished_utc": finished_utc,
                    "status": "error",
                    "artifacts_dir": str(out_dir),
                },
            )
        except Exception:
            pass
        raise
