from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from dash import dash_table, html


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.to_dict("records")


def make_table(
    *,
    table_id: str,
    df: Optional[pd.DataFrame],
    page_size: int = 15,
) -> dash_table.DataTable:
    df = df if df is not None else pd.DataFrame()
    cols = [{"name": c, "id": c} for c in df.columns.tolist()]
    return dash_table.DataTable(
        id=table_id,
        columns=cols,
        data=_df_to_records(df),
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        row_selectable="single",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "monospace",
            "fontSize": 12,
            "padding": "6px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={"fontWeight": "bold"},
    )


def card(title: str, value: Any) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontWeight": "bold", "marginBottom": "4px"}),
            html.Div(str(value)),
        ],
        style={
            "border": "1px solid #333",
            "borderRadius": "10px",
            "padding": "10px",
            "minWidth": "160px",
        },
    )
