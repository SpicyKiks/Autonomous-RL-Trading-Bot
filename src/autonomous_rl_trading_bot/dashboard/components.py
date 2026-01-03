from __future__ import annotations

from dash import html


def header(title: str, subtitle: str) -> html.Div:
    return html.Div(
        [
            html.H2(title, style={"margin": "0"}),
            html.Div(subtitle, style={"opacity": 0.8, "marginTop": "4px"}),
        ],
        style={"padding": "12px 16px", "borderBottom": "1px solid #ddd"},
    )


def empty_state(message: str) -> html.Div:
    return html.Div(
        message,
        style={
            "padding": "16px",
            "border": "1px dashed #bbb",
            "borderRadius": "10px",
            "opacity": 0.9,
        },
    )

