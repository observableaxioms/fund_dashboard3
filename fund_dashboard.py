"""
Observable Axioms — Private Credit Intelligence Platform
=========================================================
Run:   python3 fund_dashboard.py
Open:  http://127.0.0.1:8050

Install:
    pip3 install dash plotly pandas numpy
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

FUND_NAME   = "Observable Axioms — Private Credit Intelligence"
TAGLINE     = "Causal AI infrastructure for private credit transparency."
FUND_SIZE   = 50_000_000
LAUNCH_DATE = "Q2 2025"

# Private placement positions — credit lattice priced
POSITIONS = [
    {"name": "BPO Corp",             "notional": 5_000_000,  "spread_bps": 280, "protection": "buy",  "tenor": 3, "distress_score": 4.5, "sector": "BPO",        "recovery": 0.20, "pd_yr1": 0.08},
    {"name": "LegacyMedia Holdings", "notional": 3_500_000,  "spread_bps": 420, "protection": "buy",  "tenor": 5, "distress_score": 4.8, "sector": "Media",      "recovery": 0.18, "pd_yr1": 0.12},
    {"name": "MidSaaS LBO Co.",      "notional": 4_000_000,  "spread_bps": 350, "protection": "buy",  "tenor": 3, "distress_score": 4.2, "sector": "SaaS",       "recovery": 0.22, "pd_yr1": 0.09},
    {"name": "StaffCo Recruitment",  "notional": 2_500_000,  "spread_bps": 510, "protection": "buy",  "tenor": 2, "distress_score": 4.9, "sector": "Staffing",   "recovery": 0.15, "pd_yr1": 0.14},
    {"name": "HealthBack RCM Inc.",  "notional": 3_000_000,  "spread_bps": 390, "protection": "buy",  "tenor": 4, "distress_score": 4.3, "sector": "Healthcare", "recovery": 0.21, "pd_yr1": 0.10},
    {"name": "LegalDoc Services",    "notional": 1_500_000,  "spread_bps": 460, "protection": "buy",  "tenor": 2, "distress_score": 4.7, "sector": "Legal",      "recovery": 0.17, "pd_yr1": 0.13},
    {"name": "InfraGov Muni Pool",   "notional": 6_000_000,  "spread_bps":  55, "protection": "sell", "tenor": 5, "distress_score": 1.2, "sector": "Municipal",  "recovery": 0.65, "pd_yr1": 0.01},
]

SUB_STRATEGIES = [
    {"name": "Private placement credit lattice", "allocation": 40, "description": "Three-branch credit lattice (r↑, r↓, default) on unrated private placements. Default branch calibrated from proprietary distress model."},
    {"name": "Short-biased equity",              "allocation": 25, "description": "Short overvalued equity in AI-disrupted sectors via puts and direct shorts."},
    {"name": "Market neutral",                   "allocation": 20, "description": "Long AI-native disruptors, short indebted incumbents within same sector."},
    {"name": "Convertible bond arb",             "allocation": 10, "description": "Exploit mispricings in convertibles of disrupted-sector companies."},
    {"name": "Global macro",                     "allocation":  5, "description": "Commodity and infrastructure plays as AI buildout proxy."},
]

KAN_FEATURES = {
    "Debt / EBITDA":             0.91,
    "Distress score (composite)":0.88,
    "Gross margin trend":        0.82,
    "Interest coverage":         0.79,
    "Revenue growth (3yr)":      0.74,
    "Credit spread momentum":    0.68,
    "Macro rate sensitivity":    0.65,
    "Equity volatility":         0.58,
    "Days cash on hand":         0.52,
    "Capex / Depreciation":      0.44,
}

# Causal macro chain: Fed → rates → spread → bond value
CAUSAL_CHAIN = [
    ("Trigger event",      "New Fed chair appointed or FOMC policy shift",         "#C8A96E"),
    ("Rate path",          "r↑ or r↓ — binomial branch on risk-free rate",         "#5B9BD5"),
    ("Spread response",    "Credit spread widens / tightens per distress score",    "#9B7FD4"),
    ("Default branch",     "λ(t) fires — terminal state, recovery × par paid out", "#E05C5C"),
    ("Bond valuation",     "Backward induction across all surviving nodes",         "#4CAF82"),
]

# ─────────────────────────────────────────────────────────────
# CALCULATIONS
# ─────────────────────────────────────────────────────────────

def credit_lattice_value(notional, spread_bps, tenor, recovery, pd_yr1, rf=0.045):
    """
    Simplified 2-period, 3-branch credit lattice.
    Each node branches: r_up, r_down, default.
    Returns par-equivalent value and expected loss.
    """
    sigma = 0.10
    r_up   = rf * np.exp(2 * sigma)
    r_down = rf * np.exp(-2 * sigma)
    spread = spread_bps / 10000
    lam    = pd_yr1

    # Year 2 terminal values (surviving nodes)
    v_hh = notional * (1 + spread) / (1 + r_up   + spread + lam)
    v_hl = notional * (1 + spread) / (1 + rf      + spread + lam)
    v_ll = notional * (1 + spread) / (1 + r_down  + spread + lam)
    v_def= notional * recovery

    # Year 1 backward induction
    v_h = (0.5 * v_hh + 0.5 * v_hl) * (1 - lam) / (1 + r_up   + spread) + lam * v_def
    v_l = (0.5 * v_hl + 0.5 * v_ll) * (1 - lam) / (1 + r_down  + spread) + lam * v_def

    # Today
    v0 = (0.5 * v_h + 0.5 * v_l) * (1 - lam) / (1 + rf + spread) + lam * v_def

    expected_loss = notional - v0
    return round(v0, 2), round(expected_loss, 2)

def nav_simulation(fund_size, months=36, base_yield=0.072):
    np.random.seed(42)
    monthly_income = fund_size * base_yield / 12
    nav = [fund_size]
    for _ in range(1, months):
        drift = nav[-1] * np.random.normal(0.0007, 0.0022)
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
PURPLE  = "#9B7FD4"

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

def metric_card(label, value, color=TEXT, sub=None):
    return html.Div([
        html.P(label, style=LABEL_STYLE),
        html.H3(value, style=mono({"color": color, "fontSize": "26px",
                                   "fontWeight": "500", "margin": "0"})),
        html.P(sub, style=mono({"color": MUTED, "fontSize": "10px", "marginTop": "4px"})) if sub else None,
    ], style={**CARD, "padding": "20px"})

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
# STATIC CHARTS
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
    margin=dict(t=10, b=20, l=210, r=20),
    xaxis=dict(range=[0, 1], tickformat=".0%", color=MUTED, gridcolor=BORDER),
    yaxis=dict(color=TEXT, gridcolor="rgba(0,0,0,0)"),
))

debt_vals = np.linspace(2, 10, 40)
disp_vals = np.linspace(1, 5,  40)
D, Disp   = np.meshgrid(debt_vals, disp_vals)
prob      = 1 / (1 + np.exp(-(0.4 * (D - 4) + 0.6 * (Disp - 2))))
heatmap_fig = go.Figure(go.Heatmap(
    z=prob, x=debt_vals, y=disp_vals,
    colorscale=[[0, BG3], [0.4, INFO], [0.7, ACCENT], [1.0, DANGER]],
    showscale=True,
    colorbar=dict(title=dict(text="Default prob.", font=dict(color=MUTED, size=11)),
                  tickfont=dict(color=MUTED, size=10), tickformat=".0%",
                  outlinewidth=0, bgcolor="rgba(0,0,0,0)"),
    hovertemplate="D/EBITDA: %{x:.1f}x<br>Distress score: %{y:.1f}<br>PD: %{z:.0%}<extra></extra>",
))
heatmap_fig.add_shape(type="line", x0=6.5, y0=1, x1=6.5, y1=5,
                      line=dict(color=ACCENT, width=1.5, dash="dash"))
heatmap_fig.add_shape(type="line", x0=2, y0=3.5, x1=10, y1=3.5,
                      line=dict(color=ACCENT, width=1.5, dash="dash"))
heatmap_fig.add_annotation(x=8.2, y=4.5, text="High conviction<br>default zone",
                            font=dict(color=ACCENT, size=11, family="DM Mono, monospace"),
                            showarrow=False, bgcolor="rgba(11,12,14,0.7)",
                            bordercolor=ACCENT, borderwidth=1)
heatmap_fig.update_layout(**base_layout(
    margin=dict(t=10, b=50, l=60, r=20),
    xaxis=dict(title="Debt / EBITDA (x)", color=MUTED, gridcolor=BORDER),
    yaxis=dict(title="Distress score", color=MUTED, gridcolor=BORDER),
))

alloc_fig = go.Figure(go.Pie(
    labels=[s["name"] for s in SUB_STRATEGIES],
    values=[s["allocation"] for s in SUB_STRATEGIES],
    hole=0.6,
    marker=dict(colors=[ACCENT, INFO, SUCCESS, PURPLE, MUTED],
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
    legend=dict(orientation="v", x=1.02, y=0.5, font=dict(color=TEXT, size=10)),
))

# ─────────────────────────────────────────────────────────────
# SECTIONS
# ─────────────────────────────────────────────────────────────

thesis_section = html.Div([
    html.Div(style=CARD, children=[
        html.P("CORE THESIS", style=LABEL_STYLE),
        html.H2(
            "Symbolic AI can reduce the computational irreducibility Wolfram describes — "
            "and this unlocks a solution to the transparency problem in private credit.",
            style=mono({"color": ACCENT, "fontSize": "15px", "fontWeight": "400",
                        "lineHeight": "1.7", "marginBottom": "28px",
                        "borderLeft": f"3px solid {ACCENT}", "paddingLeft": "16px"})
        ),
        html.Div([
            html.Div([
                html.P(title, style=mono({"color": ACCENT, "fontSize": "10px",
                                          "letterSpacing": "0.1em", "textTransform": "uppercase",
                                          "marginBottom": "4px"})),
                html.P(body,  style=mono({"color": TEXT, "fontSize": "13px",
                                          "lineHeight": "1.7", "marginBottom": "20px"})),
            ])
            for title, body in [
                ("The problem",
                 "Private credit markets lack price transparency. There are no observable traded prices "
                 "to calibrate standard models — so incumbents either rely on rating agency proxies or "
                 "don't model credit risk dynamically at all. That opacity is both a risk and an opportunity."),
                ("The framework",
                 "A replicable strategy to capture differentiated opportunities in private credit, anchored "
                 "by a default/distress probability scoring model with explicit causal relationships, and the "
                 "ability to backtest and capture those relationships in real time across macro variables "
                 "and other factors that impact bond valuation."),
                ("The model",
                 "The default/distress scoring model is already built. The next layer uses Wolfram's multiway "
                 "graph, branchial graph, entailment cones, and causal invariance structures — so that for any "
                 "macro variable affecting a bond, the full causal chain is explicit: a trigger event (e.g. a "
                 "new Fed chair), its effect on interest rates, and how those rate movements propagate through "
                 "to each component of bond valuation."),
                ("The pricing structure",
                 "The correct framework for private placements is not the standard binomial interest rate tree — "
                 "that assumes liquid, publicly traded bonds with observable market prices. The right structure is "
                 "a credit lattice with default intensity: each node branches three ways — rates up, rates down, "
                 "and default. Default is a terminal state. The default branch probability at each node is "
                 "calibrated directly from the proprietary distress scoring model."),
                ("The Wolfram connection",
                 "The three-branch lattice maps directly to Wolfram's multiway graph formalism. Default is "
                 "literally a separate causal branch — not a probability weight attached to a rate node. "
                 "This makes the causal pathway explicit, auditable, and dynamically updatable in a way "
                 "that standard models are not."),
            ]
        ]),
    ]),

    # Causal chain visualization
    html.Div(style=CARD, children=[
        html.P("CAUSAL CHAIN — MACRO TRIGGER TO BOND VALUATION", style=LABEL_STYLE),
        html.P("Each macro event fans out as a causal branch. The full chain is explicit and auditable.",
               style=mono({"color": MUTED, "fontSize": "12px", "marginBottom": "20px"})),
        html.Div(style={"display": "flex", "alignItems": "stretch", "gap": "0"}, children=[
            html.Div([
                html.Div(style={
                    "background": BG3,
                    "borderLeft": f"3px solid {color}",
                    "borderRadius": "6px",
                    "padding": "14px 18px",
                    "marginBottom": "8px",
                }, children=[
                    html.P(step, style=mono({"color": color, "fontSize": "10px",
                                             "letterSpacing": "0.1em", "textTransform": "uppercase",
                                             "marginBottom": "4px"})),
                    html.P(desc, style=mono({"color": TEXT, "fontSize": "12px", "lineHeight": "1.5"})),
                ]),
                html.Div("↓", style=mono({"color": MUTED, "fontSize": "16px",
                                          "textAlign": "center", "marginBottom": "8px"}))
                if i < len(CAUSAL_CHAIN) - 1 else html.Div(),
            ])
            for i, (step, desc, color) in enumerate(CAUSAL_CHAIN)
        ]),
    ]),

    # Target sectors
    html.Div(style=CARD, children=[
        html.P("TARGET SECTORS — PRIVATE CREDIT UNIVERSE", style=LABEL_STYLE),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"}, children=[
            html.Div(style={"background": BG3, "borderRadius": "8px", "padding": "16px",
                             "borderLeft": f"3px solid {ACCENT}"}, children=[
                html.P(sector, style=mono({"color": TEXT, "fontSize": "13px",
                                           "fontWeight": "500", "marginBottom": "4px"})),
                html.P(desc,   style=mono({"color": MUTED, "fontSize": "11px", "lineHeight": "1.6"})),
            ])
            for sector, desc in [
                ("Business Process Outsourcing", "Document processing, customer service — directly automatable. Heavy PE debt."),
                ("Mid-market SaaS (LBO debt)",   "Single-function SaaS with 2021-vintage leveraged buyout debt. Renewal rates compressing."),
                ("Legacy Media",                 "Borrowed assuming content scarcity. AI destroys that pricing power."),
                ("Staffing & Recruitment",       "Core model is human-to-job matching — one of the most directly automated functions."),
                ("Healthcare Back Office",       "Revenue cycle management, prior auth — PE-owned, high debt, high displacement."),
                ("Commodity Legal Services",     "Document review, contract analysis — utilization collapsing at LBO-backed firms."),
            ]
        ]),
    ]),
])

# ─── LATTICE SIMULATOR ───────────────────────────────────────

lattice_section = html.Div([
    html.Div(style=CARD, children=[
        html.P("CREDIT LATTICE SIMULATOR — PRIVATE PLACEMENT PRICING", style=LABEL_STYLE),
        html.P(
            "Standard binomial rate trees assume observable market prices for calibration — "
            "private placements have none. This simulator uses the three-branch credit lattice: "
            "rates up, rates down, and default. The default branch probability feeds directly from "
            "the distress scoring model.",
            style=mono({"color": MUTED, "fontSize": "12px",
                        "lineHeight": "1.6", "marginBottom": "24px"})
        ),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "32px"}, children=[
            # Controls
            html.Div([
                html.Div([
                    html.P(label, style=mono({"color": TEXT, "fontSize": "12px", "marginBottom": "4px"})),
                    html.P(desc,  style=mono({"color": MUTED, "fontSize": "10px",
                                              "marginBottom": "8px", "lineHeight": "1.4"})),
                    dcc.Slider(id=sid, min=mn, max=mx, step=st, value=val,
                               marks={k: {"label": lbl(k),
                                          "style": {"color": MUTED, "fontSize": "10px"}}
                                      for k in marks},
                               tooltip={"placement": "top", "always_visible": False}),
                    html.Div(style={"height": "20px"}),
                ])
                for label, desc, sid, mn, mx, st, val, marks, lbl in [
                    ("Notional ($M)",     "Face value of the private placement",
                     "lat-notional", 1, 10, 0.5, 5,
                     range(1, 11, 3), lambda k: f"${k}M"),
                    ("Credit spread (bps)", "Spread over risk-free rate",
                     "lat-spread",   100, 700, 25, 350,
                     range(100, 701, 150), lambda k: str(k)),
                    ("Tenor (years)",     "Maturity of the instrument",
                     "lat-tenor",    1, 7, 1, 3,
                     range(1, 8, 2), lambda k: str(k)),
                    ("PD year 1 (%)",     "1-year default probability from distress model",
                     "lat-pd",       1, 25, 1, 8,
                     range(1, 26, 6), lambda k: f"{k}%"),
                    ("Recovery rate (%)", "Expected recovery on default",
                     "lat-recovery", 5, 60, 5, 20,
                     range(5, 61, 10), lambda k: f"{k}%"),
                ]
            ]),
            # Output
            html.Div(id="lattice-output"),
        ]),
    ]),

    # Position table
    html.Div(style=CARD, children=[
        html.P("FULL POSITION TABLE — LATTICE-PRICED", style=LABEL_STYLE),
        html.Div(id="position-table"),
    ]),
])

# ─── KAN MODEL ───────────────────────────────────────────────

kan_section = html.Div([
    html.Div(style=CARD, children=[
        html.P("KAN MODEL — KOLMOGOROV-ARNOLD NETWORK", style=LABEL_STYLE),
        html.P(
            "Unlike MLPs, KANs learn interpretable spline functions on each connection. "
            "After training, symbolic regression extracts closed-form rules — not opaque weights. "
            "Each rule is a legible investment thesis: e.g. 'default probability spikes when "
            "D/EBITDA > 6.5x AND distress score > 3.5'. Causal invariance across macro regimes "
            "is then verified using the entailment graph structure.",
            style=mono({"color": MUTED, "fontSize": "13px",
                        "lineHeight": "1.7", "marginBottom": "24px"})
        ),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px",
                         "marginBottom": "24px"}, children=[
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
                ("Input features", "45–60",  INFO,    "Financial + market + distress vectors per company-quarter"),
                ("Hidden layers",  "2 × 32", ACCENT,  "Spline activations, L1 regularised — auto-prunes irrelevant features"),
                ("Output classes", "4",      SUCCESS, "Healthy → Early distress → Severe distress → Default"),
                ("Causal branches","3/node", PURPLE,  "r↑, r↓, default — Wolfram multiway branch per lattice node"),
            ]
        ]),
        html.Div(style={"background": BG3, "borderRadius": "8px", "padding": "20px",
                         "borderLeft": f"3px solid {PURPLE}"}, children=[
            html.P("SYMBOLIC AI + WOLFRAM CAUSAL STRUCTURE", style={**LABEL_STYLE, "marginBottom": "12px"}),
            html.P(
                "The KAN model produces symbolic rules that are fed into the causal graph layer. "
                "Using Wolfram's multiway graph formalism, each macro variable (e.g. interest rates) "
                "is modeled as a branching causal path — not a scalar input. The branchial graph "
                "tracks which causal histories are equivalent (causal invariance), allowing the system "
                "to identify which macro scenarios produce isomorphic credit outcomes. "
                "This reduces the computational irreducibility of full scenario enumeration: "
                "instead of pricing every path, the entailment cone collapses equivalent branches.",
                style=mono({"color": TEXT, "fontSize": "12px", "lineHeight": "1.8"})
            ),
        ]),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
        html.Div(style=CARD, children=[
            html.P("FEATURE IMPORTANCE", style=LABEL_STYLE),
            dcc.Graph(figure=feat_fig, config={"displayModeBar": False}, style={"height": "320px"}),
        ]),
        html.Div(style=CARD, children=[
            html.P("DEFAULT PROBABILITY SURFACE — KAN SYMBOLIC RULE", style=LABEL_STYLE),
            html.P("Dashed lines = extracted thresholds: D/EBITDA > 6.5x AND distress > 3.5",
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
                ("Ingestion + NLP",    "SEC EDGAR, earnings transcripts, credit actions. FinBERT extracts covenant language, AI competition mentions, churn signals → structured feature vectors.", INFO),
                ("KAN + Symbolic rules","Feature vectors → distress probability per company per quarter. Symbolic regression extracts causal thresholds. Causal invariance tested via entailment cones.", ACCENT),
                ("Credit lattice",     "Distress PD feeds the default branch of the three-way credit lattice. Backward induction prices each private placement without requiring observable market prices.", SUCCESS),
            ])
        ]),
    ]),
])

# ─── NAV + PORTFOLIO ─────────────────────────────────────────

portfolio_section = html.Div([
    html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)",
                     "gap": "12px", "marginBottom": "20px"}, children=[
        metric_card("Fund size (pitched)", f"${FUND_SIZE//1_000_000}M"),
        metric_card("Proj. NAV (36mo)",    f"${NAV[-1]/1e6:.2f}M",    color=SUCCESS, sub="3yr projection"),
        metric_card("Total return",        f"{((NAV[-1]/FUND_SIZE)-1)*100:.1f}%", color=SUCCESS),
        metric_card("Sharpe ratio",        f"{SHARPE:.2f}",            color=INFO),
        metric_card("Max drawdown",        f"{MAX_DD:.2f}%",           color=DANGER),
    ]),
    html.Div(style=CARD, children=[
        html.P("PROJECTED PORTFOLIO NAV — 36 MONTHS", style=LABEL_STYLE),
        dcc.Graph(figure=nav_fig, config={"displayModeBar": False}, style={"height": "260px"}),
    ]),
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
                    "borderBottom": f"1px solid {BORDER}" if i < len(SUB_STRATEGIES)-1 else "none",
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
])

# ─── SCORER ──────────────────────────────────────────────────

scorer_section = html.Div([
    html.Div(style=CARD, children=[
        html.P("AI DISPLACEMENT + CREDIT DISTRESS SCORER", style=LABEL_STYLE),
        html.P(
            "Score any private credit issuer across four displacement dimensions and four credit dimensions. "
            "The composite feeds into the KAN model and calibrates the default branch of the credit lattice.",
            style=mono({"color": MUTED, "fontSize": "12px",
                        "marginBottom": "24px", "lineHeight": "1.6"})
        ),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "32px"}, children=[
            html.Div([
                html.P("DISPLACEMENT DIMENSIONS", style={**LABEL_STYLE, "marginBottom": "16px"}),
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
                        ("Revenue exposure",     "% of revenue from AI-automatable functions within 3 years",    "score-revenue"),
                        ("Pricing power",        "Ability to hold prices vs. discounting to retain customers",   "score-pricing"),
                        ("Labor leverage",       "Unit economics dependent on human headcount at scale",         "score-labor"),
                        ("Competitive moat",     "Proprietary data = durable. Switching costs alone = fragile.", "score-moat"),
                    ]
                ]),
            ]),
            html.Div([
                html.Div(id="score-display"),
                dcc.Graph(id="score-radar", config={"displayModeBar": False},
                          style={"height": "300px"}),
            ]),
        ]),
    ]),
])

# ─────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Observable Axioms — Private Credit Intelligence"

TABS = ["Thesis", "Credit Lattice Simulator", "KAN Model", "Portfolio & NAV", "Distress Scorer"]
TAB_CONTENT = [thesis_section, lattice_section, kan_section, portfolio_section, scorer_section]

def tab_btn(label, i, active=False):
    return html.Button(label, id=f"tab-{i}", n_clicks=0, style={
        "background": ACCENT if active else "transparent",
        "color": BG if active else MUTED,
        "border": f"1px solid {BORDER}", "borderRadius": "6px",
        "padding": "8px 16px", "fontFamily": "'DM Mono', monospace",
        "fontSize": "11px", "cursor": "pointer",
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
                html.H1(FUND_NAME, style=mono({"color": TEXT, "fontSize": "20px", "fontWeight": "500"})),
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
    Output("lattice-output",  "children"),
    Output("position-table",  "children"),
    Input("lat-notional",     "value"),
    Input("lat-spread",       "value"),
    Input("lat-tenor",        "value"),
    Input("lat-pd",           "value"),
    Input("lat-recovery",     "value"),
)
def update_lattice(notional_m, spread_bps, tenor, pd_pct, recovery_pct):
    notional = (notional_m or 5) * 1_000_000
    pd       = (pd_pct or 8) / 100
    recovery = (recovery_pct or 20) / 100

    v0, exp_loss = credit_lattice_value(notional, spread_bps or 350, tenor or 3, recovery, pd)
    price_pct    = v0 / notional * 100
    color        = SUCCESS if price_pct >= 98 else ACCENT if price_pct >= 94 else DANGER

    output = html.Div([
        html.P("LATTICE OUTPUT", style=LABEL_STYLE),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
            html.Div(style={"background": BG3, "borderRadius": "8px", "padding": "16px"}, children=[
                html.P("Fair value",    style=mono({"color": MUTED, "fontSize": "10px", "marginBottom": "4px"})),
                html.P(f"${v0:,.0f}",  style=mono({"color": color, "fontSize": "22px", "fontWeight": "500"})),
            ]),
            html.Div(style={"background": BG3, "borderRadius": "8px", "padding": "16px"}, children=[
                html.P("Price (% par)", style=mono({"color": MUTED, "fontSize": "10px", "marginBottom": "4px"})),
                html.P(f"{price_pct:.2f}%", style=mono({"color": color, "fontSize": "22px", "fontWeight": "500"})),
            ]),
            html.Div(style={"background": BG3, "borderRadius": "8px", "padding": "16px"}, children=[
                html.P("Expected loss",       style=mono({"color": MUTED, "fontSize": "10px", "marginBottom": "4px"})),
                html.P(f"${exp_loss:,.0f}",   style=mono({"color": DANGER, "fontSize": "22px", "fontWeight": "500"})),
            ]),
            html.Div(style={"background": BG3, "borderRadius": "8px", "padding": "16px"}, children=[
                html.P("Default branch PD",  style=mono({"color": MUTED, "fontSize": "10px", "marginBottom": "4px"})),
                html.P(f"{pd_pct:.1f}%",     style=mono({"color": PURPLE, "fontSize": "22px", "fontWeight": "500"})),
            ]),
        ]),
        html.Div(style={"background": BG3, "borderRadius": "8px", "padding": "14px", "marginTop": "12px",
                         "borderLeft": f"3px solid {PURPLE}"}, children=[
            html.P("Three-branch lattice: r↑ node → N_HH / N_HL | r↓ node → N_HL / N_LL | Default → Recovery × par",
                   style=mono({"color": MUTED, "fontSize": "10px", "lineHeight": "1.6"})),
        ]),
    ])

    # Position table
    rows = []
    for p in POSITIONS:
        v0_p, el_p = credit_lattice_value(
            p["notional"], p["spread_bps"], p["tenor"], p["recovery"], p["pd_yr1"]
        )
        rows.append({
            "Position":       p["name"],
            "Sector":         p["sector"],
            "Notional ($)":   f"${p['notional']:,}",
            "Spread (bps)":   p["spread_bps"],
            "PD yr1 (%)":     f"{p['pd_yr1']*100:.0f}%",
            "Recovery (%)":   f"{p['recovery']*100:.0f}%",
            "Fair Value ($)": f"${v0_p:,.0f}",
            "Exp. Loss ($)":  f"${el_p:,.0f}",
            "Distress":       p["distress_score"],
        })
    df = pd.DataFrame(rows)

    table = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": BG3, "color": TEXT,
                     "border": f"1px solid {BORDER}", "fontSize": "12px",
                     "padding": "10px 14px", "fontFamily": "DM Mono, monospace",
                     "textAlign": "left"},
        style_header={"backgroundColor": BG2, "color": ACCENT, "fontWeight": "500",
                       "border": f"1px solid {BORDER}", "fontSize": "11px",
                       "textTransform": "uppercase", "letterSpacing": "0.06em"},
        style_data_conditional=[
            {"if": {"filter_query": "{Distress} > 4"}, "color": DANGER},
            {"if": {"filter_query": "{Distress} < 2"}, "color": SUCCESS},
        ],
    )
    return output, table

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
    label = ("HIGH RISK — calibrate default branch"  if composite >= 4.0 else
             "MODERATE — add to watchlist"           if composite >= 3.0 else
             "LOW — insufficient displacement signal")

    display = html.Div([
        html.P("COMPOSITE DISTRESS SCORE", style=LABEL_STYLE),
        html.H2(f"{composite:.1f} / 5.0",
                style=mono({"color": color, "fontSize": "36px",
                             "fontWeight": "500", "marginBottom": "4px"})),
        html.P(label, style=mono({"color": color, "fontSize": "11px", "letterSpacing": "0.08em"})),
        html.Div(style={"marginTop": "16px", "background": BG3, "borderRadius": "8px",
                         "padding": "12px", "borderLeft": f"3px solid {PURPLE}"}, children=[
            html.P(f"Default branch PD (indicative): ~{composite*2:.0f}%",
                   style=mono({"color": PURPLE, "fontSize": "11px"})),
            html.P("Feed this into the credit lattice simulator to price the placement.",
                   style=mono({"color": MUTED, "fontSize": "10px", "marginTop": "4px"})),
        ]),
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

# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 56)
    print(f"  Observable Axioms — Private Credit Intelligence")
    print("═" * 56)
    print("  Open http://127.0.0.1:8050 in your browser")
    print("  Press Ctrl+C to stop")
    print("═" * 56 + "\n")
    app.run(debug=False)