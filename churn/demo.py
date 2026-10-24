"""Create explicitly synthetic banking records for local UI and pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(path: Path, rows: int = 600) -> Path:
    rng = np.random.default_rng(20261024)
    age = rng.integers(20, 80, size=rows)
    balance = rng.normal(78000, 52000, size=rows).clip(0)
    active = rng.binomial(1, 0.55, size=rows)
    geography = rng.choice(["France", "Germany", "Spain"], size=rows, p=[0.5, 0.27, 0.23])
    products = rng.choice([1, 2, 3], size=rows, p=[0.52, 0.42, 0.06])
    logit = -2.8 + 0.042 * (age - 40) + 0.000006 * balance + 0.65 * (products == 3) - 0.9 * active + 0.45 * (geography == "Germany")
    probability = 1 / (1 + np.exp(-logit))
    frame = pd.DataFrame({
        "CustomerId": [f"DEMO-{index:05d}" for index in range(rows)],
        "CreditScore": rng.integers(350, 850, size=rows),
        "Geography": geography,
        "Gender": rng.choice(["Female", "Male"], size=rows),
        "Age": age,
        "Tenure": rng.integers(0, 11, size=rows),
        "Balance": balance.round(2),
        "NumOfProducts": products,
        "HasCrCard": rng.binomial(1, 0.7, size=rows),
        "IsActiveMember": active,
        "EstimatedSalary": rng.uniform(10000, 200000, size=rows).round(2),
        "Exited": rng.binomial(1, probability),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/banking-demo-synthetic.csv"))
    args = parser.parse_args()
    print(make_demo_data(args.output))


if __name__ == "__main__":
    main()
