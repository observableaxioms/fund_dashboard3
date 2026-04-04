"""
KAN CDS Relationship Demo
=========================
Tests whether a tiny Kolmogorov-Arnold Network can learn the
two key CDS distress rules discovered in the fund dashboard:

    Rule 1: distress spikes when D/EBITDA > 6.5x
    Rule 2: distress spikes when displacement_score > 3.5
    Rule 3: BOTH together = high-conviction short

We generate synthetic data that encodes these rules,
train a minimal KAN, then visualise what it learned.

Install:
    pip install torch matplotlib numpy
    pip install git+https://github.com/KindXiaoming/pykan.git

Run:
    python kan_cds_demo.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – swap to "TkAgg" if you want a live window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─────────────────────────────────────────────────
# 1.  SYNTHETIC DATA
#     6 features per company-quarter observation
#     Label = distress probability (0 → 1)
# ─────────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

N_TRAIN = 800
N_TEST  = 200

def make_data(n):
    """
    Features:
        x0  D/EBITDA          uniform [1, 12]
        x1  displacement score uniform [1, 5]
        x2  interest coverage  uniform [1, 10]   (noise feature)
        x3  spread bps         uniform [50, 600]  (noise feature)
        x4  net exposure ($M)  uniform [1, 20]    (mild signal)
        x5  gross notional ($M)uniform [5, 50]    (noise feature)

    Ground-truth distress probability (what we want the KAN to recover):
        p = sigmoid(
              2.5 * (x0/6.5 - 1)          # D/EBITDA threshold at 6.5x
            + 3.0 * (x1/3.5 - 1)          # displacement threshold at 3.5
            + 1.5 * (x0/6.5 - 1)*(x1/3.5 - 1)  # interaction term
            + 0.3 * (x4/10 - 1)           # mild net-exposure signal
        )
    """
    x0 = np.random.uniform(1, 12, n)
    x1 = np.random.uniform(1,  5, n)
    x2 = np.random.uniform(1, 10, n)
    x3 = np.random.uniform(50, 600, n)
    x4 = np.random.uniform(1, 20, n)
    x5 = np.random.uniform(5, 50, n)

    logit = (
        2.5 * (x0 / 6.5 - 1)
      + 3.0 * (x1 / 3.5 - 1)
      + 1.5 * (x0 / 6.5 - 1) * (x1 / 3.5 - 1)
      + 0.3 * (x4 / 10 - 1)
    )
    p = 1 / (1 + np.exp(-logit))
    # add small noise so it's not trivially learnable
    p = np.clip(p + np.random.normal(0, 0.04, n), 0, 1)

    X = np.stack([x0, x1, x2, x3, x4, x5], axis=1).astype(np.float32)
    y = p.astype(np.float32)
    return X, y

X_tr, y_tr = make_data(N_TRAIN)
X_te, y_te = make_data(N_TEST)

# Normalise inputs to [0, 1] (KAN splines work best on bounded input)
X_min = X_tr.min(axis=0)
X_max = X_tr.max(axis=0)
X_tr_n = (X_tr - X_min) / (X_max - X_min)
X_te_n = (X_te - X_min) / (X_max - X_min)

dataset = {
    "train_input":  torch.tensor(X_tr_n),
    "train_label":  torch.tensor(y_tr).unsqueeze(1),
    "test_input":   torch.tensor(X_te_n),
    "test_label":   torch.tensor(y_te).unsqueeze(1),
}

# ─────────────────────────────────────────────────
# 2.  BUILD & TRAIN KAN
# ─────────────────────────────────────────────────

try:
    from kan import KAN
    HAS_PYKAN = True
except ImportError:
    HAS_PYKAN = False
    print("\n[INFO] pykan not installed – running fallback MLP for shape comparison.\n"
          "       Install with: pip install git+https://github.com/KindXiaoming/pykan.git\n")

FEATURE_NAMES = ["D/EBITDA", "Disp. score", "Int. coverage",
                 "Spread bps", "Net exposure", "Gross notional"]

if HAS_PYKAN:
    # width=[6, 4, 1] → 6 inputs, 4 hidden spline nodes, 1 output
    # grid=5 knots, k=3 (cubic B-spline)
    model = KAN(width=[6, 4, 1], grid=5, k=3, seed=SEED)

    # Try new API first (pykan >= 0.2), fall back to old API
    try:
        results = model.fit(
            dataset,
            opt="LBFGS",
            steps=60,
            lamb=0.001,
            lamb_entropy=2.0,
            loss_fn=torch.nn.MSELoss(),
        )
    except AttributeError:
        results = model.train(
            dataset,
            steps=60,
            lamb=0.001,
            lamb_entropy=2.0,
            loss_fn=torch.nn.MSELoss(),
        )

    train_loss = results["train_loss"]
    test_loss  = results["test_loss"]
    print(f"\nFinal train MSE : {train_loss[-1]:.5f}")
    print(f"Final test  MSE : {test_loss[-1]:.5f}")

    # Try symbolic extraction
    try:
        model.auto_symbolic(lib=["x", "x^2", "sigmoid", "tanh", "abs"])
        print("\nSymbolic rules extracted successfully.")
        model.print_node()
    except Exception as e:
        print(f"\nSymbolic extraction skipped: {e}")

else:
    # ── Fallback: tiny MLP with same architecture shape ──────
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(6, 4), nn.Tanh(),
                nn.Linear(4, 1), nn.Sigmoid(),
            )
        def forward(self, x):
            return self.net(x)

    model_mlp = MLP()
    opt = torch.optim.Adam(model_mlp.parameters(), lr=3e-3)
    loss_fn = torch.nn.MSELoss()
    train_loss, test_loss = [], []

    for step in range(300):
        model_mlp.train()
        pred = model_mlp(dataset["train_input"])
        loss = loss_fn(pred, dataset["train_label"])
        opt.zero_grad(); loss.backward(); opt.step()

        model_mlp.eval()
        with torch.no_grad():
            tl = loss_fn(model_mlp(dataset["test_input"]), dataset["test_label"]).item()
        train_loss.append(loss.item())
        test_loss.append(tl)

    print(f"\nFallback MLP – Final train MSE : {train_loss[-1]:.5f}")
    print(f"Fallback MLP – Final test  MSE : {test_loss[-1]:.5f}")


# ─────────────────────────────────────────────────
# 3.  VISUALISE
# ─────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("#0B0C0E")
GS = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

GOLD   = "#C8A96E"
TEAL   = "#4CAF82"
DANGER = "#E05C5C"
INFO   = "#5B9BD5"
MUTED  = "#6B7280"
WHITE  = "#E8E9EC"
BG2    = "#111316"
BG3    = "#181A1F"

def style_ax(ax, title):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#252830")
    ax.set_title(title, color=GOLD, fontsize=9, pad=8, loc="left")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

# ── Plot 1: training curve ──────────────────────────────────
ax0 = fig.add_subplot(GS[0, 0])
epochs = range(len(train_loss))
ax0.plot(epochs, train_loss, color=TEAL,  linewidth=1.4, label="train")
ax0.plot(epochs, test_loss,  color=GOLD,  linewidth=1.4, label="test",  linestyle="--")
ax0.set_xlabel("step"); ax0.set_ylabel("MSE")
ax0.legend(fontsize=8, facecolor=BG3, edgecolor=MUTED,
           labelcolor=WHITE, framealpha=0.8)
style_ax(ax0, "Training curve")

# ── Plot 2: D/EBITDA marginal effect ───────────────────────
ax1 = fig.add_subplot(GS[0, 1])
x0_sweep = np.linspace(1, 12, 120).astype(np.float32)
X_sweep = np.zeros((120, 6), dtype=np.float32)
X_sweep[:, 0] = x0_sweep
X_sweep[:, 1] = 3.0    # displacement held at neutral
X_sweep[:, 2] = 5.0    # int. coverage neutral
X_sweep[:, 4] = 10.0   # net exposure neutral
X_sweep_n = (X_sweep - X_min) / (X_max - X_min)
X_t = torch.tensor(X_sweep_n)

with torch.no_grad():
    if HAS_PYKAN:
        p_hat = model(X_t).squeeze().numpy()
    else:
        p_hat = model_mlp(X_t).squeeze().numpy()

# True curve
logit_true = 2.5 * (x0_sweep / 6.5 - 1) + 3.0 * (3.0 / 3.5 - 1)
p_true = 1 / (1 + np.exp(-logit_true))

ax1.plot(x0_sweep, p_true, color=MUTED,  linewidth=1.2, linestyle="--", label="ground truth")
ax1.plot(x0_sweep, p_hat,  color=GOLD,   linewidth=1.8, label="KAN learned")
ax1.axvline(6.5, color=DANGER, linewidth=0.8, linestyle=":", alpha=0.7)
ax1.text(6.6, 0.05, "6.5x threshold", color=DANGER, fontsize=7)
ax1.set_xlabel("D/EBITDA"); ax1.set_ylabel("P(distress)")
ax1.legend(fontsize=8, facecolor=BG3, edgecolor=MUTED, labelcolor=WHITE, framealpha=0.8)
ax1.set_ylim(0, 1)
style_ax(ax1, "D/EBITDA marginal effect")

# ── Plot 3: displacement score marginal effect ──────────────
ax2 = fig.add_subplot(GS[0, 2])
x1_sweep = np.linspace(1, 5, 120).astype(np.float32)
X_sweep2 = np.zeros((120, 6), dtype=np.float32)
X_sweep2[:, 0] = 5.0    # D/EBITDA held at neutral
X_sweep2[:, 1] = x1_sweep
X_sweep2[:, 2] = 5.0
X_sweep2[:, 4] = 10.0
X_sweep2_n = (X_sweep2 - X_min) / (X_max - X_min)
X_t2 = torch.tensor(X_sweep2_n)

with torch.no_grad():
    if HAS_PYKAN:
        p_hat2 = model(X_t2).squeeze().numpy()
    else:
        p_hat2 = model_mlp(X_t2).squeeze().numpy()

logit_true2 = 2.5 * (5.0 / 6.5 - 1) + 3.0 * (x1_sweep / 3.5 - 1)
p_true2 = 1 / (1 + np.exp(-logit_true2))

ax2.plot(x1_sweep, p_true2, color=MUTED, linewidth=1.2, linestyle="--", label="ground truth")
ax2.plot(x1_sweep, p_hat2,  color=INFO,  linewidth=1.8, label="KAN learned")
ax2.axvline(3.5, color=DANGER, linewidth=0.8, linestyle=":", alpha=0.7)
ax2.text(3.55, 0.05, "3.5 threshold", color=DANGER, fontsize=7)
ax2.set_xlabel("Displacement score"); ax2.set_ylabel("P(distress)")
ax2.legend(fontsize=8, facecolor=BG3, edgecolor=MUTED, labelcolor=WHITE, framealpha=0.8)
ax2.set_ylim(0, 1)
style_ax(ax2, "Displacement score marginal effect")

# ── Plot 4: joint heatmap (D/EBITDA × displacement) ────────
ax3 = fig.add_subplot(GS[1, :2])
d_vals  = np.linspace(1, 12, 60).astype(np.float32)
di_vals = np.linspace(1,  5, 60).astype(np.float32)
D, DI   = np.meshgrid(d_vals, di_vals)
X_grid  = np.zeros((60*60, 6), dtype=np.float32)
X_grid[:, 0] = D.ravel()
X_grid[:, 1] = DI.ravel()
X_grid[:, 2] = 5.0
X_grid[:, 4] = 10.0
X_grid_n = (X_grid - X_min) / (X_max - X_min)

with torch.no_grad():
    if HAS_PYKAN:
        P_hat = model(torch.tensor(X_grid_n)).squeeze().numpy().reshape(60, 60)
    else:
        P_hat = model_mlp(torch.tensor(X_grid_n)).squeeze().numpy().reshape(60, 60)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "cds", ["#111316", "#5B9BD5", "#C8A96E", "#E05C5C"])
im = ax3.contourf(D, DI, P_hat, levels=30, cmap=cmap)
cb = fig.colorbar(im, ax=ax3, fraction=0.025, pad=0.02)
cb.ax.tick_params(colors=MUTED, labelsize=8)
cb.set_label("P(distress)", color=MUTED, fontsize=8)

ax3.axvline(6.5, color=GOLD, linewidth=1.0, linestyle="--", alpha=0.8)
ax3.axhline(3.5, color=GOLD, linewidth=1.0, linestyle="--", alpha=0.8)
ax3.text(6.6, 1.1, "D/EBITDA=6.5x", color=GOLD, fontsize=7)
ax3.text(1.1, 3.55, "Disp=3.5", color=GOLD, fontsize=7)
ax3.set_xlabel("D/EBITDA"); ax3.set_ylabel("Displacement score")
style_ax(ax3, "Learned joint surface — D/EBITDA × Displacement")

# ── Plot 5: feature importance proxy (gradient-based) ──────
ax4 = fig.add_subplot(GS[1, 2])
X_ref = torch.tensor(X_te_n[:100], requires_grad=False)
X_ref.requires_grad_(True)
if HAS_PYKAN:
    out = model(X_ref)
else:
    out = model_mlp(X_ref)
out.sum().backward()
importance = X_ref.grad.abs().mean(dim=0).detach().numpy()
importance = importance / importance.max()

colors = [GOLD if v > 0.7 else INFO if v > 0.4 else MUTED for v in importance]
bars = ax4.barh(FEATURE_NAMES, importance, color=colors, height=0.6)
ax4.set_xlim(0, 1.15)
ax4.axvline(0.7, color=GOLD,  linewidth=0.6, linestyle=":", alpha=0.5)
ax4.axvline(0.4, color=MUTED, linewidth=0.6, linestyle=":", alpha=0.4)
for bar, val in zip(bars, importance):
    ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", color=WHITE, fontsize=8)
style_ax(ax4, "Feature importance (gradient)")

# ── Super-title ─────────────────────────────────────────────
fig.suptitle("KAN — CDS distress relationship learning",
             color=WHITE, fontsize=13, fontweight="normal", y=0.98)

out_path = Path("kan_cds_output.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nPlot saved → {out_path.resolve()}")
plt.close(fig)

# ─────────────────────────────────────────────────
# 4.  PRINT THRESHOLD SCAN
#     Sweep D/EBITDA at displacement=4.5 (high risk)
#     to see where the KAN fires
# ─────────────────────────────────────────────────

print("\n── Threshold scan: displacement=4.5, sweeping D/EBITDA ──")
print(f"{'D/EBITDA':>10}  {'P(distress)':>12}  {'Signal':>8}")
print("─" * 36)
for d_val in [3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0]:
    X_pt = np.zeros((1, 6), dtype=np.float32)
    X_pt[0, 0] = d_val
    X_pt[0, 1] = 4.5
    X_pt[0, 2] = 5.0
    X_pt[0, 4] = 10.0
    X_pt_n = (X_pt - X_min) / (X_max - X_min)
    with torch.no_grad():
        if HAS_PYKAN:
            p = model(torch.tensor(X_pt_n)).item()
        else:
            p = model_mlp(torch.tensor(X_pt_n)).item()
    signal = "HIGH CONVICTION" if p > 0.75 else "WATCHLIST" if p > 0.45 else "—"
    print(f"{d_val:>10.1f}  {p:>12.3f}  {signal:>8}")