from __future__ import annotations

from pathlib import Path

import pytest

from autonomous_rl_trading_bot.evaluation.reporting import (
    build_repro_payload,
    generate_run_report,
)


def test_generate_run_report_creates_all_artifacts(tmp_path: Path) -> None:
    """Test that generate_run_report creates all required artifacts."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic equity data (10 points)
    equity_rows = []
    base_equity = 1000.0
    for i in range(10):
        equity_rows.append({
            "open_time_ms": int(i * 1000),
            "equity": float(base_equity + i * 10),
            "step": i,
        })
    
    # Create synthetic trades (3 trades)
    trades_rows = [
        {
            "open_time_ms": 1000,
            "side": "BUY",
            "qty_base": 1.0,
            "price": 100.0,
            "notional": 100.0,
            "fee": 0.1,
            "realized_pnl": 5.0,
        },
        {
            "open_time_ms": 5000,
            "side": "SELL",
            "qty_base": 1.0,
            "price": 105.0,
            "notional": 105.0,
            "fee": 0.105,
            "realized_pnl": 4.895,
        },
        {
            "open_time_ms": 8000,
            "side": "BUY",
            "qty_base": 1.0,
            "price": 102.0,
            "notional": 102.0,
            "fee": 0.102,
            "realized_pnl": -2.102,
        },
    ]
    
    # Simple metrics
    metrics = {
        "final_equity": 1010.0,
        "total_return": 0.01,
        "max_drawdown": 0.05,
        "sharpe": 0.5,
    }
    
    # Reproducibility payload
    repro = build_repro_payload(
        seed=42,
        dataset_id="test_dataset",
        dataset_hash="abc123def456",
    )
    
    # Generate report
    artifact_paths = generate_run_report(
        run_dir,
        title="Test Report",
        equity_rows=equity_rows,
        trades_rows=trades_rows,
        metrics=metrics,
        repro=repro,
    )
    
    # Verify all artifacts exist
    assert (run_dir / "equity_curve.png").exists()
    assert (run_dir / "drawdown.png").exists()
    assert (run_dir / "metrics.md").exists()
    assert (run_dir / "report.html").exists()
    assert (run_dir / "report.pdf").exists()
    assert (run_dir / "repro.json").exists()
    
    # Verify artifact_paths dict
    assert "equity_curve" in artifact_paths
    assert "drawdown" in artifact_paths
    assert "metrics_md" in artifact_paths
    assert "report_html" in artifact_paths
    assert "report_pdf" in artifact_paths
    assert "repro_json" in artifact_paths
    
    # Verify HTML contains reproducibility section
    html_content = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Reproducibility" in html_content
    assert "abc123def456" in html_content  # dataset_hash
    
    # Verify repro.json contains dataset_hash
    import json
    repro_data = json.loads((run_dir / "repro.json").read_text(encoding="utf-8"))
    assert repro_data["dataset_hash"] == "abc123def456"
    assert repro_data["seed"] == 42
    assert repro_data["dataset_id"] == "test_dataset"
    
    # Verify PDF is either a real PDF or fallback text
    pdf_content = (run_dir / "report.pdf").read_bytes()
    pdf_text = pdf_content[:100].decode("utf-8", errors="ignore")
    
    # Either starts with %PDF (real PDF) or contains "reportlab" (fallback)
    assert pdf_text.startswith("%PDF") or "reportlab" in pdf_text.lower()


def test_generate_run_report_with_dataframe(tmp_path: Path) -> None:
    """Test that generate_run_report works with DataFrame inputs."""
    import pandas as pd
    
    run_dir = tmp_path / "test_run_df"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame inputs
    equity_df = pd.DataFrame({
        "open_time_ms": [i * 1000 for i in range(5)],
        "equity": [1000.0 + i * 10 for i in range(5)],
    })
    
    trades_df = pd.DataFrame({
        "open_time_ms": [1000, 3000],
        "side": ["BUY", "SELL"],
        "realized_pnl": [5.0, 4.0],
    })
    
    metrics = {"final_equity": 1010.0, "total_return": 0.01}
    repro = build_repro_payload(seed=1, dataset_id="test", dataset_hash="hash123")
    
    artifact_paths = generate_run_report(
        run_dir,
        title="Test DataFrame Report",
        equity_rows=equity_df,
        trades_rows=trades_df,
        metrics=metrics,
        repro=repro,
    )
    
    assert (run_dir / "equity_curve.png").exists()
    assert (run_dir / "drawdown.png").exists()
    assert (run_dir / "report.html").exists()

