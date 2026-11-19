"""Fig. 6 and Table 4: forward/backward signal propagation in a 40-layer random MLP.

Width 512, no bias, batch of 256 standard normal inputs, weights N(0, gain/width)
with gain 1 (LeCun-normal) or 2 (He-normal). Weight seed 2026, error-signal seed 7.
No training is performed."""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from afsurvey import ACTIVATIONS

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
NAMES = ["Tanh", "ReLU", "ELU", "SELU", "Swish", "GELU", "Mish", "RTA"]
WIDTH, DEPTH, BATCH = 512, 40, 256


def propagate(name, gain):
    rng = np.random.default_rng(2026)
    Ws = [rng.standard_normal((WIDTH, WIDTH)) * np.sqrt(gain / WIDTH) for _ in range(DEPTH)]
    h = rng.standard_normal((BATCH, WIDTH))
    f, d = ACTIVATIONS[name]
    pre, stds = [], []
    for W in Ws:
        a = h @ W; pre.append(a); h = f(a); stds.append(float(h.std()))
    g = np.random.default_rng(7).standard_normal((BATCH, WIDTH)); g0 = np.linalg.norm(g)
    grads = [0.0] * DEPTH
    for l in reversed(range(DEPTH)):
        g = g * d(pre[l]); grads[l] = float(np.linalg.norm(g) / g0); g = g @ Ws[l].T
    return dict(std=stds, grad=grads)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {init: {n: propagate(n, gain) for n in NAMES} for init, gain in [("lecun", 1.0), ("he", 2.0)]}
    json.dump(res, open(os.path.join(OUT, "signal_propagation.json"), "w"))
    with open(os.path.join(OUT, "initialisation_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["activation", "std_last_lecun", "grad_first_lecun", "std_last_he", "grad_first_he"])
        for n in NAMES:
            w.writerow([n, *(f"{res[i][n][k][idx]:.6e}" for i in ["lecun", "he"] for k, idx in [("std", -1), ("grad", 0)])])
    for n in NAMES:
        L, H = res["lecun"][n], res["he"][n]
        print(f"{n:6s} LeCun std={L['std'][-1]:.3e} grad={L['grad'][0]:.3e} | He std={H['std'][-1]:.3e} grad={H['grad'][0]:.3e}")


if __name__ == "__main__":
    main()
