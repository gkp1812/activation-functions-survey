"""Table 3: statistics of each activation for z ~ N(0, 1), by adaptive quadrature."""
import csv, json, os, sys
import numpy as np
from scipy import integrate
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from afsurvey import ACTIVATIONS

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def expect(g):
    pdf = lambda z: np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    a, _ = integrate.quad(lambda z: g(np.array(z)) * pdf(z), -np.inf, 0, limit=200)
    b, _ = integrate.quad(lambda z: g(np.array(z)) * pdf(z), 0, np.inf, limit=200)
    return float(a + b)


def main():
    os.makedirs(OUT, exist_ok=True)
    xs = np.linspace(-30, 30, 600001)
    rows = {}
    for name, (f, d) in ACTIVATIONS.items():
        mean = expect(f)
        rows[name] = dict(mean=mean, var=expect(lambda z: f(z) ** 2) - mean ** 2,
                          grad=expect(d), grad2=expect(lambda z: d(z) ** 2),
                          d0=float(d(np.array(1e-9))), fmin=float(f(xs).min()))
    json.dump(rows, open(os.path.join(OUT, "gaussian_stats.json"), "w"), indent=1)
    with open(os.path.join(OUT, "gaussian_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["activation", "mean", "variance", "mean_grad", "mean_sq_grad", "grad_at_0", "min_value_on_[-30,30]"])
        for n, r in rows.items():
            w.writerow([n] + [f"{r[k]:.6f}" for k in ["mean", "var", "grad", "grad2", "d0", "fmin"]])
    for n, r in rows.items():
        print(f"{n:11s} mean={r['mean']:+.3f} var={r['var']:.3f} g={r['grad']:.3f} q={r['grad2']:.3f} f'(0)={r['d0']:.3f}")


if __name__ == "__main__":
    main()
