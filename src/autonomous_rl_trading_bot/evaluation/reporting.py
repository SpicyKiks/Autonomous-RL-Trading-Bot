from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


def write_equity_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_trades_csv(path: Path, rows: List[Mapping[str, Any]], include_pnl: bool = True) -> None:
    """
    Write trade ledger CSV with full details including PnL.
    
    Args:
        path: Output file path
        rows: List of trade dictionaries
        include_pnl: If True, compute and include PnL column
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # still create file with header
        header = "trade_id,open_time_ms,side,qty_base,price,notional,fee,slippage_cost,pnl,reason,cash_after,qty_after,equity_after,notes\n"
        path.write_text(header, encoding="utf-8")
        return

    # Enhance rows with PnL if requested
    enhanced_rows = []
    prev_equity: Optional[float] = None
    
    for i, r in enumerate(rows):
        row_dict = dict(r)
        
        # Compute PnL
        if include_pnl:
            if "realized_pnl" in row_dict:
                pnl = float(row_dict["realized_pnl"])
            else:
                curr_equity = float(row_dict.get("equity_after", 0.0))
                if prev_equity is not None:
                    pnl = curr_equity - prev_equity
                else:
                    pnl = 0.0
                prev_equity = curr_equity
            row_dict["pnl"] = pnl
        
        # Add notes field if missing
        if "notes" not in row_dict:
            reason = str(row_dict.get("reason", "")).lower()
            side = str(row_dict.get("side", "")).upper()
            notes_parts = []
            if reason:
                notes_parts.append(f"reason:{reason}")
            if side in ("SELL", "CLOSE", "FLAT"):
                notes_parts.append("closing")
            row_dict["notes"] = "; ".join(notes_parts) if notes_parts else ""
        
        enhanced_rows.append(row_dict)
    
    # Ensure consistent field order
    preferred_order = [
        "trade_id", "open_time_ms", "side", "qty_base", "price", "notional",
        "fee", "slippage_cost", "pnl", "reason", "cash_after", "qty_after",
        "equity_after", "notes",
    ]
    
    fieldnames = []
    for field in preferred_order:
        if field in enhanced_rows[0]:
            fieldnames.append(field)
    
    # Add any remaining fields
    for field in enhanced_rows[0].keys():
        if field not in fieldnames:
            fieldnames.append(field)
    
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in enhanced_rows:
            w.writerow(r)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
