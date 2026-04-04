"""
AI Credit Displacement Fund — Strategy Dashboard
=================================================
Run: python3 fund_dashboard.py
Then open: http://127.0.0.1:8050 in your browser

Install dependencies first:
    pip3 install dash plotly pandas numpy
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime

# ─────────────────────────────────────────────────────────────
# FUND CONFIGURATION — swap in your real data here
# ─────────────────────────────────────────────────────────────

FUND_NAME   = "AI Credit Displacement Fund"
TAGLINE     = "Identifying AI-driven credit displacement before the market does."
FUND_SIZE   = 50_000_000
LAUNCH_DATE = "Q2 2025"

CDS_POSITIONS = [
    {"name": "BPO Corp",             "notional": 5_000_000, "spread_bps": 280, "protection": "buy",  "tenor": 3, "displacement": 4.5, "sector": "BPO"},
    {"name": "LegacyMedia Holdings", "notional": 3_500_000, "spread_bps": 420, "protection": "buy",  "tenor": 5, "displacement": 4.8, "sector": "Media"},
    {"name": "MidSaaS LBO Co.",      "notional": 4_000_000, "spread_bps": 350, "protection": "buy",  "tenor": 3, "displacement": 4.2, "sector": "SaaS"},
    {"name": "StaffCo Recruitment",  "notional": 2_500_000, "spread_bps": 510, "protection": "buy",  "tenor": 2, "displacement": 4.9, "sector": "Staffing"},
    {"name": "HealthBack RCM Inc.",  "notional": 3_000_000, "spread_bps": 390, "protection": "buy",  "tenor": 4, "displacement": 4.3, "sector": "Healthcare"},
    {"name": "LegalDoc Services",    "notional": 1_500_000, "spread_bps": 460, "protection": "buy",  "tenor": 2, "displacement": 4.7, "sector": "Legal"},
    {"name": "InfraGov Muni Pool",   "notional": 6_000_000, "spread_bps":  55, "protection": "sell", "tenor": 5, "displacement": 1.2, "sector": "Municipal"},
]

SUB_STRATEGIES = [
    {"name": "CDS protection (buy)",  "allocation": 35, "description": "Buy CDS on highly leveraged companies with high displacement scores"},
    {"name": "Short-biased equity",   "allocation": 30, "description": "Short overvalued equity in AI-disrupted sectors via puts and direct shorts"},
    {"name": "Market neutral",        "allocation": 20, "description": "Long AI-native disruptors, short indebted incumbents within same sector"},
    {"name": "Convertible bond arb",  "allocation": 10, "description": "Exploit mispricings in convertibles of disrupted-sector companies"},
    {"name": "Global macro",          "allocation":  5, "description": "Copper and power infrastructure as AI buildout commodity plays"},
]

KAN_FEATURES = {
    "Debt / EBITDA":        0.91,
    "Displacement score":   0.88,
    "Gross margin trend":   0.82,
    "Interest coverage":    0.79,
    "Revenue growth (3yr)": 0.74,
    "Short interest %":     0.68,
    "CDS spread momentum":  0.63,
    "Equity volatility":    0.58,
    "Days cash on hand":    0.52,
    "Capex / Depreciation": 0.44,
}

# ─────────────────────────────────────────────────────────────
# CALCULATIONS
# ─────────────────────────────────────────────────────────────

def cds_pnl(notional, spread_bps, tenor, protection, recovery):
    premium = notional * spread_bps / 10000
    if protection == "buy":
        return round(notional * (1 - recovery) - premium * tenor), round(-premium * tenor)
    else:
        return round(-notional * (1 - recovery) + premium * tenor), round(premium * tenor)

def nav_simulation(fund_size, months=36, base_yield=0.065):
    np.random.seed(42)
    monthly_income = fund_size * base_yield / 12
    nav = [fund_size]
    for _ in range(1, months):
        drift = nav[-1] * np.random.normal(0.0006, 0.0025)
        nav.append(nav[-1] + monthly_income + drift)
    returns = pd.Series(nav).pct_change().dropna()
    sharpe  = (returns.mean() / returns.std()) * np.sqrt(12)
    max_dd  = ((pd.Series(nav) / pd.Series(nav).cummax()) - 1).min()
    return nav, round(sharpe, 2), round(max_dd * 100, 2)

NAV, SHARPE, MAX_DD = nav_simulation(FUND_SIZE)
DATES = pd.date_range(start=datetime.date.today(), periods=36, freq="ME")

# ─────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────

BG      = "#0B0C0E"
BG2     = "#111316"
BG3     = "#181A1F"
BORDER  = "#252830"
TEXT    = "#E8E9EC"
MUTED   = "#6B7280"
ACCENT  = "#C8A96E"
DANGER  = "#E05C5C"
SUCCESS = "#4CAF82"
INFO    = "#5B9BD5"

CARD = {"background": BG2, "border": f"1px solid {BORDER}",
        "borderRadius": "12px", "padding": "28px", "marginBottom": "20px"}

LABEL_STYLE = {"color": MUTED, "fontSize": "11px", "letterSpacing": "0.08em",
               "textTransform": "uppercase", "marginBottom": "8px",
               "fontFamily": "'DM Mono', monospace"}

def mono(extra=None):
    base = {"fontFamily": "'DM Mono', monospace"}
    if extra:
        base.update(extra)
    return base

def metric_card(label, value, color=TEXT):
    return html.Div([
        html.P(label, style=LABEL_STYLE),
        html.H3(value, style=mono({"color": color, "fontSize": "26px",
                                   "fontWeight": "500", "margin": "0"})),
    ], style={**CARD, "padding": "20px"})

# THE FIX: base_layout is a function — each call returns a fresh dict
# so there's never a duplicate keyword argument
def base_layout(margin=None, xaxis=None, yaxis=None, **kwargs):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="DM Mono, monospace", size=11),
        legend=dict(font=dict(color=TEXT, size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=BG3, bordercolor=BORDER,
                        font_color=TEXT, font_family="DM Mono, monospace"),
        margin=margin or dict(t=10, b=40, l=60, r=20),
        xaxis=xaxis or dict(color=MUTED, gridcolor=BORDER),
        yaxis=yaxis or dict(color=MUTED, gridcolor=BORDER),
    )
    layout.update(kwargs)
    return layout

# ─────────────────────────────────────────────────────────────
# PRE-BUILD STATIC CHARTS
# ─────────────────────────────────────────────────────────────

nav_fig = go.Figure()
nav_fig.add_trace(go.Scatter(
    x=DATES, y=NAV, mode="lines", fill="tozeroy",
    line=dict(color=ACCENT, width=2),
    fillcolor="rgba(200,169,110,0.08)",
    hovertemplate="<b>%{x|%b %Y}</b><br>NAV: $%{y:,.0f}<extra></extra>",
))
nav_fig.update_layout(**base_layout(
    margin=dict(t=10, b=40, l=80, r=20),
    yaxis=dict(color=MUTED, gridcolor=BORDER, tickformat="$,.0f"),
    hovermode="x unified",
))

feat_names  = list(KAN_FEATURES.keys())
feat_vals   = list(KAN_FEATURES.values())
feat_colors = [ACCENT if v > 0.8 else INFO if v > 0.6 else MUTED for v in feat_vals]

feat_fig = go.Figure(go.Bar(
    y=feat_names, x=feat_vals, orientation="h",
    marker=dict(color=feat_colors, line=dict(width=0)),
    hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>",
))
feat_fig.update_layout(**base_layout(
    margin=dict(t=10, b=20, l=180, r=20),
    xaxis=dict(range=[0, 1], tickformat=".0%", color=MUTED, gridcolor=BORDER),
    yaxis=dict(color=TEXT, gridcolor="rgba(0,0,0,0)"),
))

debt_vals = np.linspace(2, 10, 40)
disp_vals = np.linspace(1,  5, 40)
D, Disp   = np.meshgrid(debt_vals, disp_vals)
prob      = 1 / (1 + np.exp(-(0.4 * (D - 4) + 0.6 * (Disp - 2))))

heatmap_fig = go.Figure(go.Heatmap(
    z=prob, x=debt_vals, y=disp_vals,
    colorscale=[[0, BG3], [0.4, INFO], [0.7, ACCENT], [1.0, DANGER]],
    showscale=True,
    colorbar=dict(title=dict(text="Distress prob.", font=dict(color=MUTED, size=11)),
                  tickfont=dict(color=MUTED, size=10), tickformat=".0%",
                  outlinewidth=0, bgcolor="rgba(0,0,0,0)"),
    hovertemplate="D/EBITDA: %{x:.1f}x<br>Displacement: %{y:.1f}<br>Prob: %{z:.0%}<extra></extra>",
))
heatmap_fig.add_shape(type="line", x0=6.5, y0=1, x1=6.5, y1=5,
                      line=dict(color=ACCENT, width=1.5, dash="dash"))
heatmap_fig.add_shape(type="line", x0=2, y0=3.5, x1=10, y1=3.5,
                      line=dict(color=ACCENT, width=1.5, dash="dash"))
heatmap_fig.add_annotation(x=8.2, y=4.5, text="High conviction<br>short zone",
                            font=dict(color=ACCENT, size=11, family="DM Mono, monospace"),
                            showarrow=False, bgcolor="rgba(11,12,14,0.7)",
                            bordercolor=ACCENT, borderwidth=1)
heatmap_fig.update_layout(**base_layout(
    margin=dict(t=10, b=50, l=60, r=20),
    xaxis=dict(title="Debt / EBITDA (x)", color=MUTED, gridcolor=BORDER),
    yaxis=dict(title="Displacement score", color=MUTED, gridcolor=BORDER),
))

alloc_fig = go.Figure(go.Pie(
    labels=[s["name"] for s in SUB_STRATEGIES],
    values=[s["allocation"] for s in SUB_STRATEGIES],
    hole=0.6,
    marker=dict(colors=[ACCENT, INFO, SUCCESS, "#9B7FD4", MUTED],
                line=dict(color=BG, width=2)),
    textfont=dict(color=TEXT, size=11, family="DM Mono, monospace"),
    hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>",
))
alloc_fig.add_annotation(text=f"${FUND_SIZE//1_000_000}M<br>AUM", x=0.5, y=0.5,
                          font=dict(color=TEXT, size=14, family="DM Mono, monospace"),
                          showarrow=False)
alloc_fig.update_layout(**base_layout(
    margin=dict(t=10, b=10, l=10, r=10),
    showlegend=True,
    legend=dict(orientation="v", x=1, y=0.5, font=dict(color=TEXT, size=11)),
))

cds_table_rows = []
for p in CDS_POSITIONS:
    d_pnl, nd_pnl = cds_pnl(p["notional"], p["spread_bps"], p["tenor"], p["protection"], 0.20)
    cds_table_rows.append({
        "Position":           p["name"],
        "Sector":             p["sector"],
        "Notional ($)":       f"${p['notional']:,}",
        "Spread (bps)":       p["spread_bps"],
        "Displ. Score":       p["displacement"],
        "Protection":         p["protection"].title(),
        "Default P&L ($)":    d_pnl,
        "No Default P&L ($)": nd_pnl,
    })
cds_df = pd.DataFrame(cds_table_rows)

# ─────────────────────────────────────────────────────────────
# SECTIONS
# ─────────────────────────────────────────────────────────────

thesis_section = html.Div([
    html.Div(style=CARD, children=[
        html.P("FUND THESIS", style=LABEL_STYLE),
        html.H2(TAGLINE, style=mono({"color": ACCENT, "fontSize": "18px", "fontWeight": "400",
                                     "lineHeight": "1.6", "marginBottom": "28px",
                                     "borderLeft": f"3px solid {ACCENT}", "paddingLeft": "16px"})),
        html.Div([
            html.Div([
                html.P(title, style=mono({"color": ACCENT, "fontSize": "10px", "letterSpacing": "0.1em",
                                          "textTransform": "uppercase", "marginBottom": "4px"})),
                html.P(body,  style=mono({"color": TEXT, "fontSize": "13px",
                                          "lineHeight": "1.7", "marginBottom": "20px"})),
            ])
            for title, body in [
                ("The gap",        "AI displacement is showing up in revenue figures and renewal rates. Credit markets still price these companies on historical fundamentals. That lag is where the alpha lives."),
                ("The signal",     "Every covered company receives a 4-dimension displacement vector: revenue exposure, pricing power, labor leverage, and competitive moat quality — scored 1 to 5."),
                ("The instrument", "Where debt is large enough for a liquid CDS market: buy protection. Where it isn't: short equity or buy puts. The instrument follows liquidity and timing."),
                ("The edge",       "The KAN model produces interpretable symbolic rules — not opaque weights. A rule like 'distress spikes when D/EBITDA > 6.5x AND displacement > 3.5' is a tradeable thesis you can pressure-test."),
                ("The moat",       "The displacement scoring methodology deepens as we cover more companies. Each override logged forces articulation of what the model missed — compounding intellectual edge."),
            ]
        ]),
    ]),
    html.Div(style=CARD, children=[
        html.P("DISPLACEMENT UNIVERSE — TARGET SECTORS", style=LABEL_STYLE),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"}, children=[
            html.Div(style={"background": BG3, "borderRadius": "8px", "padding": "16px",
                             "borderLeft": f"3px solid {ACCENT}"}, children=[
                html.P(sector, style=mono({"color": TEXT, "fontSize": "13px",
                                           "fontWeight": "500", "marginBottom": "4px"})),
                html.P(desc,   style=mono({"color": MUTED, "fontSize": "11px", "lineHeight": "1.6"})),
            ])
            for sector, desc in [
                ("Business Process Outsourcing", "Document processing, customer service, data entry — directly automatable within 2–3 years"),
                ("Mid-market SaaS (LBO debt)",   "Single-function SaaS with 2021-vintage PE debt; renewal rates compressing"),
                ("Legacy Media",                 "Borrowed assuming content scarcity; AI destroys that pricing power industry-wide"),
                ("Staffing & Recruitment",       "Core model is matching humans to jobs — one of the most directly automated functions"),
                ("Healthcare Back Office",       "Revenue cycle management, prior auth, medical coding — PE-owned, heavy debt"),
                ("Commodity Legal Services",     "Document review, contract analysis — utilization rates collapsing at LBO-backed firms"),
            ]
        ]),
    ]),
])

scorer_section = html.Div([
    html.Div(style=CARD, children=[
        html.P("AI DISPLACEMENT SIGNAL SCORER", style=LABEL_STYLE),
        html.P("Score any company across the four displacement dimensions. "
               "The composite feeds into the KAN model alongside traditional credit features.",
               style=mono({"color": MUTED, "fontSize": "12px",
                           "marginBottom": "24px", "lineHeight": "1.6"})),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "32px"}, children=[
            html.Div([
                html.Div([
                    html.P(label, style=mono({"color": TEXT, "fontSize": "12px", "marginBottom": "4px"})),
                    html.P(desc,  style=mono({"color": MUTED, "fontSize": "10px",
                                              "marginBottom": "8px", "lineHeight": "1.5"})),
                    dcc.Slider(id=sid, min=1, max=5, step=0.5, value=3,
                               marks={i: {"label": str(i),
                                          "style": {"color": MUTED, "fontSize": "10px"}}
                                      for i in range(1, 6)},
                               tooltip={"placement": "top", "always_visible": False}),
                    html.Div(style={"height": "20px"}),
                ])
                for label, desc, sid in [
                    ("Revenue exposure to AI",   "% of revenue from AI-automatable functions within 3 years",           "score-revenue"),
                    ("Pricing power trajectory", "Ability to hold or raise prices vs. discounting to retain customers", "score-pricing"),
                    ("Labor leverage",           "Unit economics dependent on human headcount at scale",                "score-labor"),
                    ("Competitive moat quality", "Proprietary data/regulation = durable. Switching costs alone = fragile.", "score-moat"),
                ]
            ]),
            html.Div([
                html.Div(id="score-display"),
                dcc.Graph(id="score-radar", config={"displayModeBar": False},
                          style={"height": "280px"}),
            ]),
        ]),
    ]),
])

cds_section = html.Div([
    html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)",
                     "gap": "12px", "marginBottom": "20px"}, children=[
        metric_card("Fund size (pitched)", f"${FUND_SIZE//1_000_000}M"),
        metric_card("Proj. NAV (36mo)",    f"${NAV[-1]/1e6:.2f}M",  color=SUCCESS),
        metric_card("Total return",        f"{((NAV[-1]/FUND_SIZE)-1)*100:.1f}%", color=SUCCESS),
        metric_card("Sharpe ratio",        f"{SHARPE:.2f}", color=INFO),
        metric_card("Max drawdown",        f"{MAX_DD:.2f}%", color=DANGER),
    ]),
    html.Div(style=CARD, children=[
        html.P("PROJECTED PORTFOLIO NAV — 36 MONTHS", style=LABEL_STYLE),
        dcc.Graph(figure=nav_fig, config={"displayModeBar": False}, style={"height": "260px"}),
    ]),
    html.Div(style=CARD, children=[
        html.P("CDS DEFAULT SCENARIO — ADJUST RECOVERY RATE", style=LABEL_STYLE),
        html.P("Slide to change assumed recovery rate. Green = profit if default occurs, red = loss.",
               style=mono({"color": MUTED, "fontSize": "12px",
                           "marginBottom": "16px", "lineHeight": "1.6"})),
        html.Div(style={"display": "flex", "alignItems": "center",
                         "gap": "16px", "marginBottom": "20px"}, children=[
            html.P("Recovery rate:", style=mono({"color": TEXT, "fontSize": "12px",
                                                  "whiteSpace": "nowrap"})),
            dcc.Slider(id="recovery-slider", min=0, max=60, step=5, value=20,
                       marks={i: {"label": f"{i}%",
                                  "style": {"color": MUTED, "fontSize": "10px"}}
                              for i in range(0, 65, 10)}),
        ]),
        dcc.Graph(id="cds-bar", config={"displayModeBar": False}, style={"height": "300px"}),
    ]),
    html.Div(style=CARD, children=[
        html.P("FULL POSITION TABLE — 20% RECOVERY", style=LABEL_STYLE),
        dash_table.DataTable(
            data=cds_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in cds_df.columns],
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": BG3, "color": TEXT,
                         "border": f"1px solid {BORDER}", "fontSize": "12px",
                         "padding": "10px 14px", "fontFamily": "DM Mono, monospace",
                         "textAlign": "left"},
            style_header={"backgroundColor": BG2, "color": ACCENT, "fontWeight": "500",
                           "border": f"1px solid {BORDER}", "fontSize": "11px",
                           "textTransform": "uppercase", "letterSpacing": "0.06em"},
            style_data_conditional=[
                {"if": {"filter_query": "{Default P&L ($)} > 0"}, "color": SUCCESS},
                {"if": {"filter_query": "{Default P&L ($)} < 0"}, "color": DANGER},
                {"if": {"filter_query": "{Displ. Score} > 4"},    "color": ACCENT},
            ],
        ),
    ]),
])

kan_section = html.Div([
    html.Div(style=CARD, children=[
        html.P("KAN MODEL — KOLMOGOROV-ARNOLD NETWORK", style=LABEL_STYLE),
        html.P("Unlike MLPs, KANs learn interpretable spline functions on each connection. "
               "After training, symbolic regression extracts closed-form rules — not opaque weights. "
               "Each rule is a legible investment thesis you can pressure-test and explain to a counterparty.",
               style=mono({"color": MUTED, "fontSize": "13px",
                           "lineHeight": "1.7", "marginBottom": "24px"})),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px"}, children=[
            html.Div(style={"background": BG3, "borderRadius": "8px", "padding": "16px",
                             "borderTop": f"2px solid {color}"}, children=[
                html.P(title, style=mono({"color": color, "fontSize": "10px",
                                          "letterSpacing": "0.1em", "textTransform": "uppercase",
                                          "marginBottom": "6px"})),
                html.P(val,   style=mono({"color": TEXT, "fontSize": "20px",
                                          "fontWeight": "500", "marginBottom": "4px"})),
                html.P(desc,  style=mono({"color": MUTED, "fontSize": "10px", "lineHeight": "1.5"})),
            ])
            for title, val, color, desc in [
                ("Input features", "45–60",  INFO,    "Financial + market + displacement vectors per company-quarter"),
                ("Hidden layers",  "2 × 32", ACCENT,  "Spline activations, L1 regularised — auto-prunes irrelevant features"),
                ("Output classes", "4",      SUCCESS, "Healthy → Early distress → Severe distress → Default"),
                ("Grid knots",     "5–10",   MUTED,   "Start at 5; increase to 10 if underfitting on validation set"),
            ]
        ]),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
        html.Div(style=CARD, children=[
            html.P("FEATURE IMPORTANCE (ILLUSTRATIVE)", style=LABEL_STYLE),
            dcc.Graph(figure=feat_fig, config={"displayModeBar": False}, style={"height": "320px"}),
        ]),
        html.Div(style=CARD, children=[
            html.P("SYMBOLIC RULE SURFACE — DISTRESS PROBABILITY", style=LABEL_STYLE),
            html.P("Dashed lines = discovered thresholds: D/EBITDA > 6.5x AND displacement > 3.5",
                   style=mono({"color": MUTED, "fontSize": "11px",
                               "marginBottom": "12px", "lineHeight": "1.5"})),
            dcc.Graph(figure=heatmap_fig, config={"displayModeBar": False}, style={"height": "300px"}),
        ]),
    ]),
    html.Div(style=CARD, children=[
        html.P("THREE-LAYER PIPELINE", style=LABEL_STYLE),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"}, children=[
            html.Div(style={"background": BG3, "borderRadius": "8px", "padding": "20px",
                             "borderLeft": f"3px solid {color}"}, children=[
                html.P(f"Layer {i+1}", style=mono({"color": MUTED, "fontSize": "10px",
                                                    "letterSpacing": "0.1em",
                                                    "textTransform": "uppercase",
                                                    "marginBottom": "6px"})),
                html.P(name, style=mono({"color": TEXT, "fontSize": "14px",
                                         "fontWeight": "500", "marginBottom": "8px"})),
                html.P(desc, style=mono({"color": MUTED, "fontSize": "11px", "lineHeight": "1.6"})),
            ])
            for i, (name, desc, color) in enumerate([
                ("Ingestion",               "Seeking Alpha, SEC EDGAR, earnings transcripts, rating actions. Weekly cycle into DuckDB feature store.", INFO),
                ("NLP / Signal extraction", "FinBERT extracts debt figures, covenant language, AI competition mentions, churn signals → structured feature vectors.", ACCENT),
                ("KAN + Symbolic rules",    "Feature vectors → distress probability per company per quarter. Symbolic regression extracts tradeable rule thresholds.", SUCCESS),
            ])
        ]),
    ]),
])

substrat_section = html.Div([
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
        html.Div(style=CARD, children=[
            html.P("CAPITAL ALLOCATION", style=LABEL_STYLE),
            dcc.Graph(figure=alloc_fig, config={"displayModeBar": False}, style={"height": "320px"}),
        ]),
        html.Div(style=CARD, children=[
            html.P("SUB-STRATEGIES", style=LABEL_STYLE),
            html.Div([
                html.Div(style={
                    "display": "flex", "alignItems": "flex-start", "gap": "14px",
                    "marginBottom": "16px", "paddingBottom": "16px",
                    "borderBottom": f"1px solid {BORDER}" if i < len(SUB_STRATEGIES) - 1 else "none",
                }, children=[
                    html.Div(f"{s['allocation']}%",
                             style=mono({"color": ACCENT, "fontSize": "20px",
                                         "fontWeight": "500", "minWidth": "44px"})),
                    html.Div([
                        html.P(s["name"],        style=mono({"color": TEXT, "fontSize": "13px",
                                                              "fontWeight": "500", "marginBottom": "4px"})),
                        html.P(s["description"], style=mono({"color": MUTED, "fontSize": "11px",
                                                              "lineHeight": "1.6"})),
                    ]),
                ])
                for i, s in enumerate(SUB_STRATEGIES)
            ]),
        ]),
    ]),
    html.Div(style=CARD, children=[
        html.P("VALIDATION METHODOLOGY", style=LABEL_STYLE),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
                         "gap": "12px", "marginBottom": "20px"}, children=[
            html.Div(style={"background": BG3, "borderRadius": "8px",
                             "padding": "16px", "textAlign": "center"}, children=[
                html.P(period, style=mono({"color": ACCENT, "fontSize": "14px",
                                           "fontWeight": "500", "marginBottom": "4px"})),
                html.P(label,  style=mono({"color": TEXT, "fontSize": "12px", "marginBottom": "4px"})),
                html.P(desc,   style=mono({"color": MUTED, "fontSize": "10px", "lineHeight": "1.5"})),
            ])
            for period, label, desc in [
                ("Pre-2021",     "Training set",   "Historical credit events, financials, proxy displacement scores"),
                ("2021–2022",    "Validation set", "Model selection, hyperparameter tuning, early-stop criteria"),
                ("2023–present", "Test set",       "Post-ChatGPT — when displacement started appearing in actual revenue figures"),
            ]
        ]),
        html.Div(style={"display": "flex", "gap": "32px", "paddingTop": "16px",
                         "borderTop": f"1px solid {BORDER}"}, children=[
            html.Div([
                html.P("PRIMARY METRIC", style=LABEL_STYLE),
                html.P("Precision @ top decile",
                       style=mono({"color": TEXT, "fontSize": "13px"})),
                html.P("Of the companies flagged highest risk, what fraction entered distress?",
                       style=mono({"color": MUTED, "fontSize": "11px",
                                   "lineHeight": "1.5", "marginTop": "4px"})),
            ]),
            html.Div([
                html.P("SECONDARY METRIC", style=LABEL_STYLE),
                html.P("AUROC (full distribution)",
                       style=mono({"color": TEXT, "fontSize": "13px"})),
                html.P("Not predicting every default — finding a concentrated high-conviction short set.",
                       style=mono({"color": MUTED, "fontSize": "11px",
                                   "lineHeight": "1.5", "marginTop": "4px"})),
            ]),
        ]),
    ]),
])

# ─────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = f"{FUND_NAME} — Strategy Overview"

TABS        = ["Thesis", "Displacement Scorer", "CDS Simulator", "KAN Model", "Sub-strategies"]
TAB_CONTENT = [thesis_section, scorer_section, cds_section, kan_section, substrat_section]

def tab_btn(label, i, active=False):
    return html.Button(label, id=f"tab-{i}", n_clicks=0, style={
        "background": ACCENT if active else "transparent",
        "color": BG if active else MUTED,
        "border": f"1px solid {BORDER}", "borderRadius": "6px",
        "padding": "8px 16px", "fontFamily": "'DM Mono', monospace",
        "fontSize": "12px", "cursor": "pointer",
        "letterSpacing": "0.04em", "whiteSpace": "nowrap",
    })

app.layout = html.Div(style={"background": BG, "minHeight": "100vh", "color": TEXT}, children=[
    html.Div(style={"background": BG2, "borderBottom": f"1px solid {BORDER}",
                     "padding": "24px 48px"}, children=[
        html.Div(style={"display": "flex", "alignItems": "flex-start",
                         "justifyContent": "space-between", "marginBottom": "20px"}, children=[
            html.Div([
                html.P("CONFIDENTIAL — NOT FOR DISTRIBUTION",
                       style=mono({"color": DANGER, "fontSize": "10px",
                                   "letterSpacing": "0.12em", "marginBottom": "6px"})),
                html.H1(FUND_NAME, style=mono({"color": TEXT, "fontSize": "22px", "fontWeight": "500"})),
                html.P(TAGLINE,    style=mono({"color": MUTED, "fontSize": "12px", "marginTop": "4px"})),
            ]),
            html.Div([
                html.P(f"Launch: {LAUNCH_DATE}",
                       style=mono({"color": MUTED, "fontSize": "11px", "textAlign": "right"})),
                html.P(f"Target AUM: ${FUND_SIZE//1_000_000}M",
                       style=mono({"color": ACCENT, "fontSize": "14px",
                                   "fontWeight": "500", "textAlign": "right"})),
            ]),
        ]),
        html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}, children=[
            tab_btn(label, i, active=(i == 0)) for i, label in enumerate(TABS)
        ]),
    ]),
    html.Div(style={"padding": "32px 48px"}, children=[
        html.Div(id="tab-content", children=TAB_CONTENT[0]),
    ]),
    dcc.Store(id="active-tab", data=0),
])

# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("active-tab", "data"),
    [Input(f"tab-{i}", "n_clicks") for i in range(len(TABS))],
    prevent_initial_call=True,
)
def set_tab(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 0
    return int(ctx.triggered[0]["prop_id"].split(".")[0].split("-")[1])

@app.callback(Output("tab-content", "children"), Input("active-tab", "data"))
def render_tab(active):
    return TAB_CONTENT[active]

@app.callback(
    Output("score-display", "children"),
    Output("score-radar",   "figure"),
    Input("score-revenue",  "value"),
    Input("score-pricing",  "value"),
    Input("score-labor",    "value"),
    Input("score-moat",     "value"),
)
def update_scorer(rev, price, labor, moat):
    composite = (rev + price + labor + moat) / 4
    color = DANGER if composite >= 4.0 else ACCENT if composite >= 3.0 else SUCCESS
    label = ("HIGH RISK — build position"  if composite >= 4.0 else
             "MODERATE — add to watchlist" if composite >= 3.0 else
             "LOW — insufficient displacement")

    display = html.Div([
        html.P("COMPOSITE SCORE", style=LABEL_STYLE),
        html.H2(f"{composite:.1f} / 5.0",
                style=mono({"color": color, "fontSize": "36px",
                             "fontWeight": "500", "marginBottom": "4px"})),
        html.P(label, style=mono({"color": color, "fontSize": "11px",
                                   "letterSpacing": "0.08em"})),
    ], style={"textAlign": "center", "padding": "20px 0"})

    categories = ["Revenue exposure", "Pricing power", "Labor leverage", "Moat quality"]
    values     = [rev, price, labor, moat]
    radar = go.Figure(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(200,169,110,0.12)",
        line=dict(color=color, width=2), marker=dict(color=color, size=6),
    ))
    radar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 5], color=MUTED, gridcolor=BORDER,
                            tickfont=dict(size=9, color=MUTED, family="DM Mono, monospace")),
            angularaxis=dict(color=TEXT, gridcolor=BORDER,
                             tickfont=dict(size=10, color=TEXT, family="DM Mono, monospace")),
        ),
        margin=dict(t=20, b=20, l=40, r=40),
        showlegend=False,
    )
    return display, radar

@app.callback(Output("cds-bar", "figure"), Input("recovery-slider", "value"))
def update_cds(recovery_pct):
    recovery = recovery_pct / 100
    names, dpnls = [], []
    for p in CDS_POSITIONS:
        d_pnl, _ = cds_pnl(p["notional"], p["spread_bps"], p["tenor"], p["protection"], recovery)
        names.append(p["name"])
        dpnls.append(d_pnl)

    fig = go.Figure(go.Bar(
        y=names, x=dpnls, orientation="h",
        marker=dict(color=[SUCCESS if v > 0 else DANGER for v in dpnls], line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Default P&L: $%{x:,.0f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=BORDER, line_width=1)
    fig.update_layout(**base_layout(
        margin=dict(t=10, b=30, l=180, r=20),
        xaxis=dict(tickformat="$,.0f", color=MUTED, gridcolor=BORDER),
        yaxis=dict(color=TEXT, gridcolor="rgba(0,0,0,0)"),
    ))
    return fig

# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 52)
    print(f"  {FUND_NAME}")
    print("  Strategy Dashboard")
    print("═" * 52)
    print("  Open http://127.0.0.1:8050 in your browser")
    print("  Press Ctrl+C to stop")
    print("═" * 52 + "\n")
    app.run(debug=False)