#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python scripts/gaussian_stats.py
python scripts/signal_propagation.py
python scripts/make_figures.py
