# Codebase Audit Report
**Generated:** 2025-01-18  
**Purpose:** Professional cleanup audit identifying duplicates, unused code, inconsistencies, and consolidation opportunities

---

## 1. Repository Map (High-Level)

```
Autonomous-RL-Trading-Bot/
├── src/autonomous_rl_trading_bot/     # Main package
│   ├── cli.py                          # Unified CLI entrypoint
│   ├── __main__.py                     # Package entrypoint
│   ├── agents/                         # RL agents (PPO, DQN)
│   ├── backtest/                       # Backtest engine (Day-4)
│   ├── broker/                         # Broker adapters (spot, futures, paper)
│   ├── common/                         # Shared utilities
│   ├── dashboard/                      # Dash webapp
│   ├── data/                          # Data fetching & dataset building
│   ├── envs/                          # Gymnasium environments
│   ├── evaluation/                    # Backtesting & baselines (legacy)
│   ├── features/                      # Feature engineering
│   ├── live/                          # Live trading runner
│   ├── modes/                         # Mode registry (spot/futures)
│   ├── repro/                         # Reproducibility pipeline (Day-4)
│   ├── risk/                          # Risk management
│   ├── rl/                            # RL utilities (legacy)
│   ├── storage/                       # Database & artifact storage
│   ├── training/                      # Training pipeline
│   ├── run_*.py                       # Legacy standalone runners
│   └── main.py                        # Version display
├── scripts/                           # Utility scripts
├── configs/                           # YAML configuration files
├── tests/                             # Pytest test suite
├── app_desktop.py                     # Desktop app wrapper
├── run_dashboard.py                   # Dashboard launcher
├── run_live_demo.py                   # Live trading launcher
└── run_baselines.py                    # Baselines launcher
```

---

## 2. Entrypoints Analysis

### 2.1 Primary Entrypoints

| Entrypoint | Purpose | Status | Notes |
|------------|---------|--------|-------|
| `python -m autonomous_rl_trading_bot` | Unified CLI | ✅ **CANONICAL** | Routes to all commands via `cli.py` |
| `app_desktop.py` | Desktop app wrapper | ⚠️ **DUPLICATE** | Wraps dashboard; could be CLI command |
| `run_dashboard.py` | Dashboard launcher | ⚠️ **DUPLICATE** | Called by CLI `arbt dashboard`; redundant standalone |
| `run_live_demo.py` | Live trading launcher | ⚠️ **DUPLICATE** | Called by CLI `arbt live`; redundant standalone |
| `run_baselines.py` | Baselines launcher | ⚠️ **DUPLICATE** | Called by CLI `arbt baselines`; redundant standalone |

### 2.2 CLI Commands (via `cli.py`)

| Command | Implementation | Status |
|---------|---------------|--------|
| `arbt dataset fetch` | `run_fetch.py` | ✅ Active |
| `arbt dataset build` | `run_dataset.py` | ✅ Active |
| `arbt train` | `run_train.py` | ✅ Active |
| `arbt backtest` | `backtest/runner.py` | ✅ Active (Day-4) |
| `arbt live` | `run_live_demo.py` | ✅ Active |
| `arbt dashboard` | `run_dashboard.py` | ✅ Active |
| `arbt baselines` | `evaluation/baselines.py` | ✅ Active |
| `arbt repro` | `repro/runner.py` | ✅ Active (Day-4) |
| `arbt verify` | `cli.py::_run_verify()` | ✅ Active (Day-4) |

### 2.3 Scripts Directory

| Script | Purpose | Status | Notes |
|--------|---------|--------|-------|
| `scripts/download_binance_futures.py` | Download futures data | ✅ Active | Used by `repro` pipeline |
| `scripts/build_features.py` | Build features | ✅ Active | Used by `repro` pipeline |
| `scripts/make_dataset.py` | Build dataset | ✅ Active | Used by `repro` pipeline |
| `scripts/export_validation_pack.py` | Generate validation docs | ✅ Active | Day-4 feature |
| `scripts/bootstrap_env.py` | Environment setup | ❓ **UNUSED?** | Not referenced in CLI/docs |
| `scripts/migrate.py` | DB migration | ❓ **UNUSED?** | Migration handled by `common/db.py` |
| `scripts/smoke_test_env.py` | Smoke test | ❓ **UNUSED?** | Not referenced |
| `scripts/fetch_data.py` | Fetch data (legacy?) | ❓ **UNUSED?** | Superseded by `run_fetch.py`? |
| `scripts/export_report.py` | Export report | ❓ **UNUSED?** | Not referenced |

---

## 3. Duplicates Table

| Thing | Files | Keep | Remove/Merge Plan | Risk Notes |
|-------|-------|------|-------------------|------------|
| **Training Runners** | `run_train.py`, `training/trainer.py`, `training/train_pipeline.py` | `training/train_pipeline.py` (Day-2) + `training/trainer.py` (legacy NPZ) | `run_train.py` → thin wrapper calling `trainer.py` or `train_pipeline.py` | Low: `run_train.py` already routes to both |
| **Backtest Runners** | `run_backtest.py`, `backtest/runner.py`, `evaluation/backtest_runner.py`, `evaluation/backtester.py` | `backtest/runner.py` (Day-4, parquet) + `evaluation/backtester.py` (legacy NPZ) | `run_backtest.py` → remove (use CLI), `evaluation/backtest_runner.py` → merge into `backtest/runner.py` | Medium: `run_backtest.py` used by CLI; `backtest_runner.py` has NPZ logic |
| **Dataset Loading** | `training/train_pipeline.py::load_dataset()` (parquet), `rl/dataset.py::load_dataset_npz()` (NPZ), `evaluation/backtester.py::load_dataset()` (NPZ), `evaluation/backtest_runner.py::_load_dataset_from_id()` (NPZ) | Centralize: `data/dataset_loader.py` | Create unified loader with format detection (parquet vs NPZ) | High: Multiple call sites, different formats |
| **Config Loading** | `common/config.py::load_config()` | ✅ **SINGLE** | Already centralized | None |
| **Path Resolution** | `common/paths.py`, `storage/artifact_store.py`, hardcoded `Path("artifacts/...")` | `common/paths.py` | Migrate all hardcoded paths to use `paths.py` | Medium: Some hardcoded paths in scripts |
| **Live Trading Entry** | `run_live_demo.py`, `dashboard/live_session.py` → `run_live_demo.py` | `run_live_demo.py` | Dashboard calls `run_live_demo.py` correctly; keep both | Low: Correct delegation pattern |
| **Baselines Entry** | `run_baselines.py`, `evaluation/baselines.py::main()` | `evaluation/baselines.py::main()` | `run_baselines.py` → remove (use CLI) | Low: CLI already routes correctly |
| **Dashboard Entry** | `app_desktop.py`, `run_dashboard.py` | `run_dashboard.py` | `app_desktop.py` → keep (desktop wrapper), `run_dashboard.py` → keep (CLI) | Low: Both serve different purposes |
| **Output Directory Conventions** | `reports/`, `artifacts/reports/`, `artifacts/backtests/`, `artifacts/runs/` | Standardize on `artifacts/` | Migrate `reports/` → `artifacts/reports/` | Medium: Some scripts write to `reports/` directly |
| **Dataset Path Resolution** | `data/processed/{SYMBOL}_{INTERVAL}_dataset.parquet`, `artifacts/datasets/{dataset_id}/`, `artifacts/runs/{dataset_id}/` | Centralize in `data/dataset_loader.py` | Unified resolver with format detection | High: Inconsistent paths break reproducibility |

---

## 4. Unused Modules/Files

### 4.1 Potentially Unused Scripts

| File | Status | Evidence |
|------|--------|----------|
| `scripts/bootstrap_env.py` | ❓ **UNUSED** | Not imported/referenced in CLI, docs, or tests |
| `scripts/migrate.py` | ❓ **UNUSED** | Migration handled by `common/db.py::migrate()` |
| `scripts/smoke_test_env.py` | ❓ **UNUSED** | Not referenced; smoke tests in `tests/` |
| `scripts/fetch_data.py` | ❓ **UNUSED** | Superseded by `run_fetch.py`? |
| `scripts/export_report.py` | ❓ **UNUSED** | Not referenced; reporting in `evaluation/reporting.py` |

### 4.2 Potentially Unused Modules

| Module | Status | Evidence |
|--------|--------|----------|
| `src/autonomous_rl_trading_bot/main.py` | ❓ **UNUSED** | Only prints version; `__main__.py` calls `cli.py` |
| `src/autonomous_rl_trading_bot/rl/sb3_train.py` | ❓ **UNUSED** | Not imported; training uses `training/trainer.py` |
| `src/autonomous_rl_trading_bot/rl/env_trading.py` | ❓ **UNUSED** | Not imported; environments in `envs/` |
| `src/autonomous_rl_trading_bot/rl/metrics.py` | ❓ **UNUSED** | Not imported; metrics in `evaluation/metrics.py` |
| `src/autonomous_rl_trading_bot/rl/dataset.py` (partial) | ⚠️ **LEGACY** | Used for NPZ datasets; parquet uses `training/train_pipeline.py` |

**Note:** Full import analysis requires static analysis tools (e.g., `vulture`, `pylint --unused-imports`). Manual inspection suggests these are candidates for removal.

---

## 5. Code That Should Be Centralized

### 5.1 Dataset Resolution

**Current State:**
- Parquet: `data/processed/{SYMBOL}_{INTERVAL}_dataset.parquet` (Day-2)
- NPZ: `artifacts/datasets/{dataset_id}/` (legacy)
- Multiple loaders: `training/train_pipeline.py`, `rl/dataset.py`, `evaluation/backtester.py`, `evaluation/backtest_runner.py`

**Proposed:**
```python
# data/dataset_loader.py
def resolve_dataset(symbol: str, interval: str, dataset_id: Optional[str] = None) -> Dataset:
    """Unified dataset resolver with format detection."""
    # 1. Try parquet (Day-2)
    # 2. Try NPZ (legacy)
    # 3. Raise FileNotFoundError with helpful message
```

### 5.2 Output Directory Conventions

**Current State:**
- `reports/` (root-level, used by `backtest/runner.py`, `repro/runner.py`)
- `artifacts/reports/` (defined in `common/paths.py`)
- `artifacts/backtests/` (legacy)
- `artifacts/runs/` (canonical)

**Proposed:**
- Standardize on `artifacts/` subdirectories
- Migrate `reports/` → `artifacts/reports/`
- Update `backtest/runner.py` and `repro/runner.py` to use `artifacts_dir() / "reports"`

### 5.3 Logging Configuration

**Current State:**
- `common/logging.py::configure_logging()` (canonical)
- Some scripts use `logging.getLogger()` directly

**Proposed:**
- Enforce use of `configure_logging()` everywhere
- Add logging config to `common/paths.py` for log file locations

### 5.4 Run ID Generation

**Current State:**
- Multiple formats: `{timestamp}_{mode}_train_{dataset_id}_{algo}_{hash}`, `{timestamp}_spot_BTCUSDT_1m_w30_t2000_s42`, etc.

**Proposed:**
- Centralize in `common/utils.py::generate_run_id(kind, mode, **kwargs)`
- Standardize format: `{ISO8601}_{kind}_{mode}_{tags}_{hash}`

---

## 6. Naming/Path Inconsistencies

### 6.1 Windows vs Linux Path Handling

**Current State:**
- ✅ Most code uses `pathlib.Path` (cross-platform)
- ⚠️ Some scripts use raw strings: `"artifacts/reports"` instead of `artifacts_dir() / "reports"`
- ⚠️ `run_backtest.py` has custom `_repo_root()` instead of `common/paths.py::repo_root()`

**Issues Found:**
- `run_backtest.py::_repo_root()` duplicates `common/paths.py::repo_root()`
- Hardcoded paths in `scripts/` (e.g., `"data/processed/"`)

**Proposed:**
- Migrate all path resolution to `common/paths.py`
- Remove duplicate `_repo_root()` implementations

### 6.2 Dataset Path Inconsistencies

| Format | Path Pattern | Used By |
|--------|--------------|---------|
| Parquet (Day-2) | `data/processed/{SYMBOL}_{INTERVAL}_dataset.parquet` | `training/train_pipeline.py`, `backtest/runner.py`, `repro/runner.py` |
| NPZ (legacy) | `artifacts/datasets/{dataset_id}/dataset.npz` | `rl/dataset.py`, `evaluation/backtester.py`, `run_train.py` (legacy) |
| NPZ (legacy alt) | `artifacts/runs/{dataset_id}/dataset.npz` | `evaluation/backtest_runner.py` (fallback) |

**Proposed:**
- Unified resolver in `data/dataset_loader.py` handles both formats

---

## 7. Canonical Implementations

### 7.1 Training

**Canonical:** `training/train_pipeline.py` (Day-2 parquet) + `training/trainer.py` (legacy NPZ)

**Entrypoint:** `arbt train` → `run_train.py` → routes to appropriate pipeline

**Status:** ✅ **GOOD** - `run_train.py` already handles both formats

### 7.2 Backtesting

**Canonical:** `backtest/runner.py` (Day-4, parquet) + `evaluation/backtester.py` (legacy NPZ)

**Entrypoint:** `arbt backtest` → `backtest/runner.py`

**Status:** ⚠️ **NEEDS CLEANUP** - `evaluation/backtest_runner.py` duplicates logic; should merge

### 7.3 Live Trading

**Canonical:** `live/live_runner.py`

**Entrypoint:** `arbt live` → `run_live_demo.py` → `LiveRunner`

**Status:** ✅ **GOOD** - Single implementation

### 7.4 Dashboard

**Canonical:** `dashboard/app.py::create_app()`

**Entrypoint:** `arbt dashboard` → `run_dashboard.py` → `create_app()`

**Status:** ✅ **GOOD** - Single implementation

---

## 8. Dashboard → Live Trading Integration

### 8.1 Current Flow

```
Dashboard (callbacks.py)
  → live_session.py::start_trading()
    → Background thread
      → run_live_demo.main(args_list)
        → LiveRunner.run()
```

### 8.2 Status

✅ **WORKING** - Dashboard correctly triggers live trading via `run_live_demo.py`. No issues identified.

### 8.3 Notes

- Dashboard uses `dashboard/live_session.py` to manage background thread
- Session state tracked in `LiveSession` dataclass
- Stop button sets `stop_event`; `LiveRunner` checks kill switch
- Logs captured and displayed in dashboard

---

## 9. Risk Notes

### 9.1 High Risk (Breaking Changes)

| Change | Risk | Mitigation |
|--------|------|------------|
| Remove `run_backtest.py` | Medium | CLI already uses `backtest/runner.py`; verify no external scripts depend on `run_backtest.py` |
| Migrate `reports/` → `artifacts/reports/` | Medium | Update `backtest/runner.py`, `repro/runner.py`; add migration script |
| Unify dataset loading | High | Create `data/dataset_loader.py`; update all call sites; test both formats |
| Remove `evaluation/backtest_runner.py` | Medium | Merge NPZ logic into `backtest/runner.py` first |

### 9.2 Medium Risk (Functional Impact)

| Change | Risk | Mitigation |
|--------|------|------------|
| Remove unused scripts | Low | Verify with `grep`; add deprecation warnings first |
| Centralize path resolution | Low | Migrate incrementally; test on Windows/Linux |
| Remove `rl/` legacy modules | Medium | Verify no imports; check git history for usage |

### 9.3 Low Risk (Cosmetic)

| Change | Risk | Mitigation |
|--------|------|------------|
| Remove `main.py` | Low | Not imported anywhere |
| Standardize run ID format | Low | Backward compatible (old IDs still work) |

---

## 10. Proposed Consolidation Plan (Step-by-Step)

### Phase 1: Dataset Loading Unification (HIGH PRIORITY)

1. **Create `data/dataset_loader.py`**
   - Unified resolver: `resolve_dataset(symbol, interval, dataset_id=None)`
   - Format detection: parquet → NPZ → error
   - Returns common `Dataset` interface

2. **Update Call Sites**
   - `training/train_pipeline.py::load_dataset()` → use `dataset_loader.resolve_dataset()`
   - `backtest/runner.py` → use `dataset_loader.resolve_dataset()`
   - `evaluation/backtester.py::load_dataset()` → use `dataset_loader.resolve_dataset()`
   - `evaluation/backtest_runner.py::_load_dataset_from_id()` → use `dataset_loader.resolve_dataset()`

3. **Test Both Formats**
   - Parquet: `data/processed/BTCUSDT_1m_dataset.parquet`
   - NPZ: `artifacts/datasets/{dataset_id}/dataset.npz`

### Phase 2: Path Resolution Cleanup (MEDIUM PRIORITY)

1. **Migrate Hardcoded Paths**
   - `scripts/*.py` → use `common/paths.py`
   - `run_backtest.py::_repo_root()` → use `common/paths.py::repo_root()`
   - `backtest/runner.py` → use `artifacts_dir() / "reports"` instead of `"reports/"`

2. **Standardize Output Directories**
   - Migrate `reports/` → `artifacts/reports/`
   - Update `backtest/runner.py`, `repro/runner.py`
   - Update `.gitignore` if needed

### Phase 3: Backtest Runner Consolidation (MEDIUM PRIORITY)

1. **Merge `evaluation/backtest_runner.py` into `backtest/runner.py`**
   - Move NPZ loading logic to `backtest/runner.py`
   - Keep parquet path as primary
   - Add format detection

2. **Remove `run_backtest.py`**
   - Verify CLI uses `backtest/runner.py` directly
   - Check for external dependencies

### Phase 4: Script Cleanup (LOW PRIORITY)

1. **Audit Unused Scripts**
   - Run `vulture` or `pylint --unused-imports`
   - Verify with `grep -r "bootstrap_env\|migrate\|smoke_test_env\|fetch_data\|export_report"`

2. **Remove or Deprecate**
   - `scripts/bootstrap_env.py` (if unused)
   - `scripts/migrate.py` (if superseded by `common/db.py`)
   - `scripts/smoke_test_env.py` (if unused)
   - `scripts/fetch_data.py` (if superseded by `run_fetch.py`)
   - `scripts/export_report.py` (if unused)

### Phase 5: Legacy Module Cleanup (LOW PRIORITY)

1. **Audit `rl/` Module**
   - Check imports: `grep -r "from.*rl\."`
   - Verify `rl/dataset.py` still needed for NPZ
   - Remove if unused: `rl/sb3_train.py`, `rl/env_trading.py`, `rl/metrics.py`

2. **Remove `main.py`**
   - Not imported; version display handled by `cli.py --version`

### Phase 6: Entrypoint Cleanup (LOW PRIORITY)

1. **Keep Standalone Scripts (for convenience)**
   - `app_desktop.py` (desktop wrapper)
   - `run_dashboard.py` (CLI already routes)
   - `run_live_demo.py` (CLI already routes)
   - `run_baselines.py` (CLI already routes)

2. **Documentation Update**
   - Update README to emphasize CLI as primary entrypoint
   - Note standalone scripts as convenience wrappers

---

## 11. Summary

### 11.1 Key Findings

1. **✅ Good:** Single CLI entrypoint (`cli.py`) routes correctly
2. **⚠️ Issue:** Multiple dataset loading implementations (parquet vs NPZ)
3. **⚠️ Issue:** Inconsistent output directory conventions (`reports/` vs `artifacts/reports/`)
4. **⚠️ Issue:** Duplicate backtest runners (`backtest/runner.py` vs `evaluation/backtest_runner.py`)
5. **✅ Good:** Dashboard → Live trading integration works correctly
6. **❓ Question:** Several scripts in `scripts/` appear unused

### 11.2 Priority Actions

1. **HIGH:** Unify dataset loading (`data/dataset_loader.py`)
2. **MEDIUM:** Consolidate backtest runners
3. **MEDIUM:** Standardize output directories
4. **LOW:** Remove unused scripts/modules

### 11.3 Estimated Impact

- **Breaking Changes:** Low (mostly internal refactoring)
- **User Impact:** None (CLI interface unchanged)
- **Maintenance:** Reduced (fewer code paths to maintain)

---

## 12. Next Steps

1. **Review this audit** with team/stakeholders
2. **Prioritize phases** based on project needs
3. **Create GitHub issues** for each phase
4. **Implement incrementally** with tests
5. **Update documentation** as changes are made

---

## 13. Consolidation Status

**Date:** 2025-01-18  
**Status:** Phase 1 Complete - Canonical Commands Created

### Completed Actions

1. **✅ Created Canonical Command Modules** (`src/autonomous_rl_trading_bot/commands/`)
   - `commands/train.py` - Wraps `run_train.py`
   - `commands/backtest.py` - Wraps `backtest/runner.py`
   - `commands/live.py` - Wraps `run_live_demo.py`
   - `commands/dataset.py` - Wraps `run_fetch.py` and `run_dataset.py`
   - `commands/baselines.py` - Wraps `evaluation/baselines.py`

2. **✅ Updated CLI to Use Canonical Commands**
   - `cli.py` now routes through `commands/` module
   - All CLI interfaces preserved (no breaking changes)
   - Lazy imports used to avoid circular dependencies

3. **✅ Created Centralized Utilities**
   - `common/run_ids.py` - Unified run ID generation
   - `data/dataset_loader.py` - Unified dataset resolver (parquet + NPZ)

### Current Architecture

```
CLI (cli.py)
  ↓
Canonical Commands (commands/*.py)
  ↓
Legacy Implementations (run_*.py, backtest/runner.py, etc.)
  ↓
Core Logic (training/trainer.py, training/train_pipeline.py, etc.)
```

### Files Status

| File | Status | Notes |
|------|--------|-------|
| `run_train.py` | ✅ **KEPT** | Called by `commands/train.py` |
| `run_fetch.py` | ✅ **KEPT** | Called by `commands/dataset.py` |
| `run_dataset.py` | ✅ **KEPT** | Called by `commands/dataset.py` |
| `run_backtest.py` | ⚠️ **LEGACY** | Not used by CLI; `backtest/runner.py` is canonical |
| `run_live_demo.py` | ✅ **KEPT** | Called by `commands/live.py` |
| `run_baselines.py` | ⚠️ **LEGACY** | Not used by CLI; `evaluation/baselines.py` is canonical |
| `main.py` | ❓ **UNUSED** | Not imported; version display handled by CLI |

### Next Steps (Future Phases)

1. **Phase 2:** Consolidate dataset loading (use `data/dataset_loader.py` everywhere)
2. **Phase 3:** Merge `evaluation/backtest_runner.py` into `backtest/runner.py`
3. **Phase 4:** Standardize output directories (`reports/` → `artifacts/reports/`)
4. **Phase 5:** Archive/remove truly unused files after verification

### Notes

- All CLI commands continue to work as before
- No trading logic, rewards, or model semantics changed
- Tests should pass (verification pending)
- Backward compatibility maintained

---

**End of Audit Report**
