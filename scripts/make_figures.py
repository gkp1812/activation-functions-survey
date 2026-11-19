"""Figures 1-7 of the survey. Run gaussian_stats.py and signal_propagation.py first."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from afsurvey import ACTIVATIONS as AF

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "legend.fontsize": 7, "legend.frameon": False,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
W = 84 / 25.4  # journal column width in inches
DPI = 600
stats = json.load(open(os.path.join(ROOT, "results", "gaussian_stats.json")))
sim = json.load(open(os.path.join(ROOT, "results", "signal_propagation.json")))["lecun"]
sim_names = list(sim.keys())
depth = len(sim[sim_names[0]]["std"])

# ---------- Fig 3/4: curves and derivatives ----------
x = np.linspace(-4, 4, 801)
style = {"Sigmoid": ("-", None), "Tanh": ("--", None), "ReLU": ("-", None), "Leaky ReLU": (":", None),
         "ELU": ("-.", None), "SELU": ((0, (5, 1, 1, 1)), None), "Swish": ("-", None), "GELU": ("--", None),
         "Mish": (":", None), "LiSHT": ("-.", None), "RTA": ("-", None)}
groupA = ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "ELU", "SELU"]
groupB = ["Tanh", "Swish", "GELU", "Mish", "LiSHT", "RTA"]
cA = ["#4a4a4a", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
cB = ["#1f77b4", "#ff7f0e", "#17becf", "#7f7f7f", "#bcbd22", "#000000"]

def panel(kind, fname, ylims):
    fig, axs = plt.subplots(2, 1, figsize=(W, W * 1.25))
    for ax, grp, cols, tag in zip(axs, [groupA, groupB], [cA, cB], ["(a)", "(b)"]):
        for n, c in zip(grp, cols):
            f, d = AF[n]
            y = f(x) if kind == "f" else d(x)
            lw = 1.8 if n == "RTA" else 1.1
            ax.plot(x, y, linestyle=style[n][0], color=c, lw=lw, label=n)
        ax.axhline(0, color="0.7", lw=0.4); ax.axvline(0, color="0.7", lw=0.4)
        ax.set_xlim(-4, 4); ax.set_ylim(*ylims)
        ax.set_xlabel("Input x"); ax.set_ylabel("f(x)" if kind == "f" else "f'(x)")
        ax.legend(ncol=3, loc="upper left", handlelength=2.4)
        ax.text(0.98, 0.04, tag, transform=ax.transAxes, ha="right", fontsize=8)
        ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(f"{OUT}/{fname}", dpi=DPI); plt.close(fig)
panel("f", "fig3.png", (-2.0, 4.6))
panel("d", "fig4.png", (-0.3, 1.9))

# ---------- Fig 5: mean output and mean gradient ----------
names = ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "ELU", "SELU", "Swish", "GELU", "Mish", "LiSHT", "RTA"]
fig, ax = plt.subplots(figsize=(W, W * 0.72))
idx = np.arange(len(names)); bw = 0.38
ax.bar(idx - bw / 2, [stats[n]["mean"] for n in names], bw, color="0.35", label="Mean output E[f(z)]")
ax.bar(idx + bw / 2, [stats[n]["grad"] for n in names], bw, color="white", edgecolor="0.1",
       hatch="////", lw=0.6, label="Mean gradient E[f'(z)]")
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(idx); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_ylabel("Value for z ~ N(0, 1)"); ax.legend(loc="upper left"); ax.set_ylim(-0.1, 1.25)
ax.grid(axis="y", alpha=0.25, lw=0.4)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5.png", dpi=DPI); plt.close(fig)

# ---------- Fig 6: signal propagation (LeCun init) ----------
cs = ["#1f77b4", "#d62728", "#9467bd", "#8c564b", "#ff7f0e", "#17becf", "#7f7f7f", "#000000"]
ls = ["--", "-", "-.", (0, (5, 1, 1, 1)), ":", "--", ":", "-"]
mk = ["o", "s", "^", "v", "D", "x", "+", "*"]
fig, axs = plt.subplots(2, 1, figsize=(W, W * 1.25))
L = np.arange(1, depth + 1)
for n, c, s, m in zip(sim_names, cs, ls, mk):
    lw = 1.8 if n == "RTA" else 1.0
    axs[0].semilogy(L, sim[n]["std"], linestyle=s, color=c, lw=lw, marker=m, markevery=6, ms=3, label=n)
    axs[1].semilogy(L, sim[n]["grad"], linestyle=s, color=c, lw=lw, marker=m, markevery=6, ms=3, label=n)
axs[0].set_ylabel("Activation std"); axs[1].set_ylabel("Relative gradient norm")
for ax, t in zip(axs, ["(a)", "(b)"]):
    ax.set_xlabel("Layer index"); ax.grid(alpha=0.25, lw=0.4); ax.set_xlim(1, depth)
    ax.text(0.02, 0.04, t, transform=ax.transAxes, ha="left")
axs[1].legend(ncol=2, loc="lower right", handlelength=2.2, fontsize=6.5)
fig.tight_layout(h_pad=0.6); fig.savefig(f"{OUT}/fig6.png", dpi=DPI); plt.close(fig)

# ---------- diagram helpers ----------
def box(ax, x, y, w, h, text, fc="white", bold=False, fs=7):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc, ec="0.15", lw=0.7))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.15",
                                                                   mutation_scale=7, shrinkA=0, shrinkB=0))

# ---------- Fig 1: taxonomy ----------
fig, ax = plt.subplots(figsize=(W, W * 1.05)); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.axis("off")
box(ax, 0.5, 0.97, 0.62, 0.07, "Activation functions", fc="0.85", bold=True, fs=8)
fams = [("Saturating\n(sigmoidal)", "Sigmoid, Tanh,\nHard-Tanh, Softsign"),
        ("Rectified\n(ReLU family)", "ReLU, Leaky ReLU,\nPReLU, RReLU"),
        ("Exponential\nunits", "ELU, SELU, CELU"),
        ("Smooth\nself-gated", "Swish/SiLU, GELU,\nMish, GLU, SwiGLU"),
        ("Tanh-based\nhybrids", "LiSHT, TanhSoft,\nReLTanh, RTA"),
        ("Adaptive /\nlearnable", "APL, PAU, adaptive\nslope, KAN splines")]
ys = np.linspace(0.82, 0.08, len(fams))
ax.plot([0.08, 0.08], [0.97, ys[-1]], color="0.15", lw=0.7); ax.plot([0.08, 0.19], [0.97, 0.97], color="0.15", lw=0.7)
for (fam, ex), y in zip(fams, ys):
    arrow(ax, 0.08, y, 0.13, y)
    box(ax, 0.30, y, 0.33, 0.105, fam, fc="0.93", bold=True)
    arrow(ax, 0.47, y, 0.52, y)
    box(ax, 0.755, y, 0.46, 0.105, ex)
fig.tight_layout(pad=0.1); fig.savefig(f"{OUT}/fig1.png", dpi=DPI); plt.close(fig)

# ---------- Fig 2: timeline ----------
events = [(1986, "Sigmoid + backprop"), (1998, "Tanh (Efficient BackProp)"), (2010, "ReLU"),
          (2013, "Leaky ReLU"), (2015, "PReLU, ELU"), (2016, "GELU"), (2017, "SELU, Swish, GLU"),
          (2019, "Mish, ReLTanh, LiSHT"), (2020, "SwiGLU, PAU, TanhSoft"), (2024, "KAN (learnable edges)"),
          (2025, "RTA")]
fig, ax = plt.subplots(figsize=(W, W * 0.78)); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(-0.5, len(events) - 0.5)
ax.plot([0.27, 0.27], [-0.3, len(events) - 0.7], color="0.2", lw=1.0)
for i, (yr, ev) in enumerate(events):
    y = len(events) - 1 - i
    ax.plot(0.27, y, "o", ms=4.5, mfc="black" if ev == "RTA" else "white", mec="black", mew=0.8)
    ax.text(0.22, y, str(yr), ha="right", va="center", fontsize=7.5, fontweight="bold")
    ax.text(0.32, y, ev, ha="left", va="center", fontsize=7.5, fontweight="bold" if ev == "RTA" else "normal")
fig.tight_layout(pad=0.1); fig.savefig(f"{OUT}/fig2.png", dpi=DPI); plt.close(fig)

# ---------- Fig 9: task-dependent selection framework ----------
fig, ax = plt.subplots(figsize=(W, W * 1.12)); ax.set_xlim(0, 1); ax.set_ylim(0, 1.12); ax.axis("off")
box(ax, 0.5, 1.07, 0.70, 0.07, "Task, data and deployment profile", fc="0.85", bold=True)
q = [("Need smooth higher-order\nderivatives (PINN, ODE)?", "Tanh, adaptive Tanh,\nSine (SIREN)"),
     ("Transformer or large\nlanguage model?", "GELU, SwiGLU\n(gated FFN)"),
     ("Deep CNN for vision\nor medical images?", "ReLU baseline; Swish,\nMish, ELU, RTA"),
     ("Recurrent gates or\nbounded outputs?", "Sigmoid (gates),\nTanh (state)"),
     ("Edge / low-precision\nhardware?", "ReLU, ReLU6,\nHard-Swish")]
ys = np.linspace(0.92, 0.26, len(q))
arrow(ax, 0.30, 1.035, 0.30, ys[0] + 0.055)
for i, ((qq, ans), y) in enumerate(zip(q, ys)):
    box(ax, 0.30, y, 0.52, 0.105, qq)
    arrow(ax, 0.56, y, 0.62, y); ax.text(0.59, y + 0.008, "yes", fontsize=6, ha="center", va="bottom")
    box(ax, 0.81, y, 0.34, 0.105, ans, fc="0.93")
    if i < len(q) - 1:
        arrow(ax, 0.30, y - 0.0525, 0.30, ys[i + 1] + 0.0525)
        ax.text(0.315, (y + ys[i + 1]) / 2, "no", fontsize=6, va="center")
arrow(ax, 0.30, ys[-1] - 0.0525, 0.30, 0.135)
box(ax, 0.5, 0.09, 0.80, 0.09, "Benchmark 2-3 candidates on validation data;\nfine-tune shape parameters if learnable", fc="0.85")
fig.tight_layout(pad=0.1); fig.savefig(f"{OUT}/fig9.png", dpi=DPI); plt.close(fig)

# ---------- Fig 7: where activations sit in common architectures ----------
def vbox(ax, x, y, w, h, t, af=False, fs=7.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.005,rounding_size=0.015",
                                fc="0.25" if af else "white", ec="0.1", lw=0.7))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, color="white" if af else "black",
            fontweight="bold" if af else "normal")
panels = [
    ("(a) CNN residual block", ["Conv + BN", "ReLU / Swish / RTA", "Conv + BN", "Add skip", "ReLU / Swish / RTA"], [1, 4]),
    ("(b) Transformer FFN", ["LayerNorm", "Linear d \u2192 4d", "GELU / SwiGLU", "Linear 4d \u2192 d", "Add residual"], [2]),
    ("(c) LSTM cell", ["Concat [h, x]", "Gates f, i, o: Sigmoid", "Candidate g: Tanh", "c = f\u2299c + i\u2299g", "h = o\u2299Tanh(c)"], [1, 2, 4]),
    ("(d) PINN", ["Inputs (x, t)", "Dense + Tanh", "Dense + Tanh", "Output u(x, t)", "Autodiff u\u209c, u\u2093\u2093 in loss"], [1, 2]),
]
fig, axs = plt.subplots(2, 2, figsize=(W, W * 1.2))
for ax, (title, boxes, afidx) in zip(axs.flat, panels):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=8, fontweight="bold")
    ys = np.linspace(0.80, 0.08, len(boxes))
    for i, (b, y) in enumerate(zip(boxes, ys)):
        vbox(ax, 0.5, y, 0.94, 0.12, b, af=i in afidx, fs=7)
        if i < len(boxes) - 1:
            arrow(ax, 0.5, y - 0.06, 0.5, ys[i + 1] + 0.06)
fig.tight_layout(pad=0.2, w_pad=0.4, h_pad=0.6)
fig.savefig(f"{OUT}/fig7.png", dpi=DPI); plt.close(fig)

# ---------- Fig 8: benchmark datasets used in activation-function studies ----------
studies = [("ReLU (2011)", ["MNIST", "CIFAR-10"]), ("PReLU (2015)", ["ImageNet"]),
           ("RReLU (2015)", ["CIFAR-10", "CIFAR-100"]), ("ELU (2016)", ["MNIST", "CIFAR-10", "CIFAR-100", "ImageNet"]),
           ("GELU (2016)", ["MNIST", "CIFAR-10", "CIFAR-100", "Speech", "Text"]),
           ("Swish (2017)", ["CIFAR-10", "CIFAR-100", "ImageNet", "Text"]), ("LiSHT (2019)", ["CIFAR-100", "Text"]),
           ("Mish (2019)", ["CIFAR-10", "ImageNet", "MS-COCO"]), ("PAU (2020)", ["MNIST", "F-MNIST", "CIFAR-10", "ImageNet"]),
           ("TanhSoft (2020)", ["MNIST", "CIFAR-10", "CIFAR-100"]),
           ("Dubey et al. (2022)", ["CIFAR-10", "CIFAR-100", "Speech", "Text"]),
           ("RTA (2025)", ["F-MNIST", "CIFAR-10", "CIFAR-100", "Chest X-ray"])]
cols = ["MNIST", "F-MNIST", "CIFAR-10", "CIFAR-100", "ImageNet", "MS-COCO", "Chest X-ray", "Speech", "Text"]
M = np.array([[c in ds for c in cols] for _, ds in studies], dtype=float)
fig, ax = plt.subplots(figsize=(W, W * 0.95))
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if M[i, j]:
            ax.add_patch(plt.Rectangle((j + 0.12, i + 0.12), 0.76, 0.76, color="black" if studies[i][0].startswith("RTA") else "0.45"))
ax.set_xlim(0, len(cols)); ax.set_ylim(len(studies), -1.1)
ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_xticklabels(cols, rotation=50, ha="right")
ax.set_yticks(np.arange(len(studies)) + 0.5); ax.set_yticklabels([s for s, _ in studies])
for j, n in enumerate(M.sum(0).astype(int)):
    ax.text(j + 0.5, -0.45, str(n), ha="center", va="center", fontsize=7.5, fontweight="bold")
ax.text(-0.15, -0.45, "Count", ha="right", va="center", fontsize=7.5, fontweight="bold")
ax.set_xticks(np.arange(len(cols) + 1), minor=True); ax.set_yticks(np.arange(len(studies) + 1), minor=True)
ax.grid(which="minor", color="0.85", lw=0.5); ax.tick_params(which="minor", length=0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(pad=0.2); fig.savefig(f"{OUT}/fig8.png", dpi=DPI); plt.close(fig)

print("figures written to", os.path.abspath(OUT))
