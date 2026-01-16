from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from autonomous_rl_trading_bot.common.logging import configure_logging
from autonomous_rl_trading_bot.common.time import utc_now_compact
from autonomous_rl_trading_bot.common.utils import safe_json_dump


def _repo_root() -> Path:
    # src/autonomous_rl_trading_bot/run_backtest.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _artifacts_dir() -> Path:
    return _repo_root() / "artifacts"


def _find_latest_dataset_id(*, mode: str) -> str:
    """
    Choose the most recent dataset directory under artifacts/datasets that matches mode hint.
    This matches how your CLI behaved previously (auto-select dataset when not provided).
    """
    base = _artifacts_dir() / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"No datasets directory found at: {base}")

    mode = mode.lower().strip()
    candidates = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if mode in name:
            candidates.append(p)

    if not candidates:
        # fallback: pick any dataset dir if none match
        candidates = [p for p in base.iterdir() if p.is_dir()]

    if not candidates:
        raise FileNotFoundError(f"No dataset directories found under: {base}")

    # Pick newest by modified time
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0].name


def _resolve_policy_path(policy_arg: str, logger) -> Optional[Path]:
    """
    Robustly resolve a policy file path.
    
    Supports:
    - Exact file path (policy.zip, model.zip, etc.)
    - Directory path (scans for zip files)
    - Relative paths (resolved from repo root or cwd)
    - Windows paths (backslashes normalized)
    
    Priority order for zip files:
    1. policy.zip
    2. model.zip
    3. best_model.zip
    4. Any other *.zip file
    
    Returns:
        Path to the policy zip file, or None if not found
    """
    if not policy_arg:
        return None
    
    # Normalize path separators (handle Windows backslashes)
    policy_arg_normalized = policy_arg.replace("\\", "/")
    initial_path = Path(policy_arg_normalized)
    
    # Try to resolve the path
    candidates_to_try = []
    
    # If absolute path, use as-is
    if initial_path.is_absolute():
        candidates_to_try.append(initial_path.resolve())
    else:
        # Try relative to repo root
        candidates_to_try.append(_repo_root() / policy_arg_normalized)
        # Try relative to current working directory
        candidates_to_try.append(Path(policy_arg_normalized).resolve())
        # Try with original path (in case it's already relative to cwd)
        candidates_to_try.append(Path(policy_arg).resolve())
    
    # Try each candidate
    for candidate in candidates_to_try:
        if candidate.exists():
            # If it's a directory, scan for zip files
            if candidate.is_dir():
                result = _find_policy_zip_in_dir(candidate, logger)
                if result:
                    return result
                # Directory exists but no zip files found - continue to try other candidates
                continue
            # If it's a file and ends with .zip, use it
            elif candidate.is_file() and candidate.suffix.lower() == ".zip":
                return candidate
            # If it's a file but not .zip, that's unexpected
            elif candidate.is_file():
                logger.warning(f"Path exists but is not a zip file: {candidate}")
                continue
    
    # If none of the candidates exist as files, check if any candidate's parent is a directory
    # (e.g., user passed a file path that doesn't exist, but parent directory does)
    checked_parents = set()
    for candidate in candidates_to_try:
        parent = candidate.parent
        parent_str = str(parent.resolve())
        if parent_str in checked_parents:
            continue  # Already checked this parent
        checked_parents.add(parent_str)
        
        if parent.exists() and parent.is_dir():
            logger.info(f"Path {candidate.name} not found, checking parent directory: {parent}")
            result = _find_policy_zip_in_dir(parent, logger)
            if result:
                return result
    
    # Nothing found - return None (caller will handle error)
    return None


def _find_policy_zip_in_dir(directory: Path, logger) -> Optional[Path]:
    """
    Find a policy zip file in a directory, following priority order:
    1. policy.zip
    2. model.zip
    3. best_model.zip
    4. Any other *.zip file
    
    Returns:
        Path to the first zip file found, or None if no zip files exist
    """
    if not directory.exists() or not directory.is_dir():
        return None
    
    # Priority order
    priority_names = ["policy.zip", "model.zip", "best_model.zip"]
    
    # Try priority names first
    for name in priority_names:
        candidate = directory / name
        if candidate.exists() and candidate.is_file():
            logger.info(f"Found policy file: {candidate}")
            return candidate
    
    # If no priority file found, scan for any .zip file
    zip_files = list(directory.glob("*.zip"))
    zip_files = [z for z in zip_files if z.is_file()]
    
    if zip_files:
        # Sort by name for determinism, pick first
        zip_files.sort(key=lambda p: p.name.lower())
        logger.info(f"No priority policy file found, using: {zip_files[0]} (found {len(zip_files)} zip file(s) total)")
        return zip_files[0]
    
    # No zip files found - log what files ARE in the directory for debugging
    all_files = [f.name for f in directory.iterdir() if f.is_file()]
    if all_files:
        logger.warning(f"Directory {directory} exists but contains no zip files. Found {len(all_files)} other file(s): {', '.join(all_files[:5])}{'...' if len(all_files) > 5 else ''}")
    else:
        logger.warning(f"Directory {directory} exists but is empty")
    
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="arbt backtest", description="Run backtest evaluation (baseline + reporting artifacts).")
    parser.add_argument("--mode", choices=["spot", "futures"], required=True)
    parser.add_argument("--policy", default=None, help="Path to SB3 policy.zip file, or baseline strategy name (default: baseline). If path ends with .zip, loads SB3 model.")
    parser.add_argument("--dataset-id", default=None, help="Dataset id under artifacts/datasets/. If omitted, auto-select latest.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id. If omitted, generated.")
    args = parser.parse_args(argv)

    artifacts = _artifacts_dir()
    artifacts.mkdir(parents=True, exist_ok=True)

    dataset_id = args.dataset_id or _find_latest_dataset_id(mode=args.mode)
    dataset_dir = artifacts / "datasets" / dataset_id
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    run_id = args.run_id or f"run_backtest_{args.mode}_{utc_now_compact()}_{dataset_id[:8]}"
    run_dir = artifacts / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(run_id=run_id, run_dir=run_dir, console=True, log_level="INFO")
    logger.info("Backtest starting")
    logger.info(f"run_id={run_id} market_type={args.mode} dataset_id={dataset_id} policy={args.policy}")

    # Determine policy argument: if --policy is a path (file or directory), resolve it; otherwise treat as strategy name
    policy_arg = args.policy
    if policy_arg is None:
        policy_arg = "baseline"
    
    # Check if policy looks like a path (contains slashes/backslashes or ends with .zip)
    policy_path = None
    # Check if it's path-like: has path separators, ends with .zip, or exists as a path
    is_path_like = (
        policy_arg and 
        (policy_arg.endswith(".zip") or 
         "/" in policy_arg or 
         "\\" in policy_arg or
         Path(policy_arg).exists() or
         (_repo_root() / policy_arg.replace("\\", "/")).exists())
    )
    
    if is_path_like:
        # Try to resolve the policy path
        policy_path = _resolve_policy_path(policy_arg, logger)
        
        if policy_path is None:
            # Build helpful error message
            tried_paths = []
            policy_arg_normalized = policy_arg.replace("\\", "/")
            initial_path = Path(policy_arg_normalized)
            
            if initial_path.is_absolute():
                tried_paths.append(str(initial_path.resolve()))
            else:
                tried_paths.append(str(_repo_root() / policy_arg_normalized))
                tried_paths.append(str(Path(policy_arg_normalized).resolve()))
                tried_paths.append(str(Path(policy_arg).resolve()))
            
            # Check what zip files exist in relevant directories and what files ARE in those directories
            found_zips = []
            checked_dirs = set()
            dir_contents = {}
            
            for tried in tried_paths:
                tried_path = Path(tried)
                # Check the path itself if it's a directory
                if tried_path.exists() and tried_path.is_dir():
                    dir_str = str(tried_path.resolve())
                    if dir_str not in checked_dirs:
                        checked_dirs.add(dir_str)
                        zips = list(tried_path.glob("*.zip"))
                        found_zips.extend([str(z) for z in zips if z.is_file()])
                        # Also list what files ARE in the directory
                        all_files = [f.name for f in tried_path.iterdir() if f.is_file()]
                        if all_files:
                            dir_contents[str(tried_path)] = all_files[:20]  # Limit to first 20
                
                # Check parent directory
                parent = tried_path.parent
                parent_str = str(parent.resolve())
                if parent.exists() and parent.is_dir() and parent_str not in checked_dirs:
                    checked_dirs.add(parent_str)
                    zips = list(parent.glob("*.zip"))
                    found_zips.extend([str(z) for z in zips if z.is_file()])
                    # Also list what files ARE in the parent directory
                    all_files = [f.name for f in parent.iterdir() if f.is_file()]
                    if all_files:
                        dir_contents[str(parent)] = all_files[:20]  # Limit to first 20
            
            error_msg = f"Policy file not found: {policy_arg}\n"
            error_msg += f"\nTried paths:\n"
            for tp in tried_paths:
                tp_path = Path(tp)
                status = ""
                if tp_path.exists():
                    if tp_path.is_dir():
                        status = " (directory exists)"
                    elif tp_path.is_file():
                        status = " (file exists)"
                error_msg += f"  - {tp}{status}\n"
            
            if found_zips:
                error_msg += f"\nFound {len(found_zips)} zip file(s) in nearby directories:\n"
                for fz in found_zips[:10]:  # Limit to first 10
                    error_msg += f"  - {fz}\n"
                if len(found_zips) > 10:
                    error_msg += f"  ... and {len(found_zips) - 10} more\n"
            else:
                error_msg += "\nNo zip files found in nearby directories.\n"
            
            if dir_contents:
                error_msg += "\nFiles found in checked directories:\n"
                for dir_path, files in dir_contents.items():
                    error_msg += f"  {dir_path}:\n"
                    for fname in files:
                        error_msg += f"    - {fname}\n"
                    if len(files) == 20:
                        error_msg += f"    ... (showing first 20 files)\n"
            
            # Suggest checking for other run directories or retraining
            runs_dir = _artifacts_dir() / "runs"
            if runs_dir.exists():
                try:
                    recent_runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], 
                                        key=lambda x: x.stat().st_mtime, reverse=True)[:5]
                    if recent_runs:
                        error_msg += f"\nRecent training runs (check these for policy.zip):\n"
                        for run_dir in recent_runs:
                            try:
                                has_zip = any(f.suffix == ".zip" for f in run_dir.iterdir() if f.is_file())
                                status = "✓ has zip" if has_zip else "✗ no zip"
                                error_msg += f"  - {run_dir.name} ({status})\n"
                            except (PermissionError, OSError):
                                # Skip directories we can't read
                                pass
                        error_msg += "\nTip: Use a run directory that contains policy.zip, or retrain to generate it.\n"
                except (PermissionError, OSError) as e:
                    logger.warning(f"Could not list runs directory: {e}")
            
            raise FileNotFoundError(error_msg)
        
        policy_arg = str(policy_path)

    try:
        from autonomous_rl_trading_bot.evaluation.backtest_runner import run_backtest

        payload = run_backtest(
            mode=args.mode,
            dataset_id=dataset_id,
            run_id=run_id,
            policy=policy_arg if policy_arg != "baseline" else None,
            cfg=None,  # Let backtest_runner load cfg from dataset meta
        )

        # Mirror generated artifacts into run_dir for easy marking evidence
        from autonomous_rl_trading_bot.common.fs import copytree_merge

        produced_dir = artifacts / "backtests" / run_id
        if produced_dir.exists():
            copytree_merge(produced_dir, run_dir)

        safe_json_dump(run_dir / "run_output.json", payload)

        logger.info("Backtest DONE")
        logger.info(f"Artifacts: {run_dir}")
        return 0
    except Exception as e:
        # Ensure error is captured for marking evidence
        err_path = run_dir / "error.txt"
        err_path.write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
        logger.exception("Backtest FAILED. See error.txt")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
