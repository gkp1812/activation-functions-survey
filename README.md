# activation-functions-survey

Code and results for a survey of activation functions in deep neural networks by G.K. Pandey and S. Srivastava (Birla Institute of Technology, Mesra), submitted to the *ICTACT Journal on Data Science and Machine Learning*, 2026.

The scripts reproduce every quantitative result and figure in the paper. No external or proprietary datasets are used: all inputs are synthetic standard normal samples and randomly initialised weights generated with fixed seeds.

## What is reproduced

| Paper item | Script | Output |
|---|---|---|
| Table 3: statistics for z ~ N(0, 1) | `scripts/gaussian_stats.py` | `results/gaussian_stats.csv`, `.json` |
| Fig. 6 and Table 4: signal propagation, LeCun vs He init | `scripts/signal_propagation.py` | `results/signal_propagation.json`, `results/initialisation_summary.csv` |
| Fig. 1-9 | `scripts/make_figures.py` | `figures/fig1.png` ... `figures/fig9.png` (600 dpi) |

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh
```

Runtime is about one minute on a laptop CPU.

## Notes on the figures

* Fig. 7 (activation placement in CNN, Transformer, LSTM and PINN blocks) and Fig. 9 (selection framework) are schematic diagrams.
* Fig. 8 maps benchmark datasets to the experiments reported in the original papers of each activation function and in the benchmark of Dubey et al. (Neurocomputing, 2022); the mapping is defined in `scripts/make_figures.py`.

## Activation functions

`afsurvey/activations.py` defines each function and its derivative in NumPy: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, SELU, Swish (SiLU), GELU, Mish, LiSHT and the Rectified Tangent Activation (RTA):

```
RTA(x) = max(x, tanh(x))   # x for x >= 0, tanh(x) for x < 0
```

RTA was introduced in G.K. Pandey and S. Srivastava, IEEE Access, vol. 13, pp. 120028-120039, 2025, https://doi.org/10.1109/ACCESS.2025.3587602. A Keras/TensorFlow implementation is available on PyPI as `rt-activation`.

## Experimental settings

* **Gaussian statistics:** mean, variance, mean gradient and mean squared gradient of f(z) for z ~ N(0, 1), computed by adaptive quadrature (`scipy.integrate.quad`).
* **Signal propagation:** fully connected network, 40 layers, width 512, no bias, batch of 256 standard normal inputs. Weights are drawn from N(0, gain/512) with gain 1 (LeCun-normal) or 2 (He-normal). Weight seed 2026, error-signal seed 7. The network is not trained; the experiment measures behaviour at initialisation.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT License. See `LICENSE`.
