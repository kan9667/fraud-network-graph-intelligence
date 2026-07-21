#!/usr/bin/env python3
"""CLI: offline evaluation of unsupervised detection against synthetic ground truth.

Usage:
  python3 evaluate.py
  python3 -m src.evaluate

Ground truth is read ONLY here (and in src/evaluate.py). Detection/dashboard
remain fully unsupervised.
"""

from src.evaluate import main

if __name__ == "__main__":
    main()
