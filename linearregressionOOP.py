"""
Ontario Retail (March 2025) - Regression Data + Pattern Recognition
OOP Python script:
- Generates (or uses) a CSV dataset
- Reads it using OOP
- Runs multiple linear regression (normal equation)
- Performs basic pattern recognition (correlations + residual outliers)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import csv
import numpy as np


CSV_FILENAME = "ontario_retail_march_2025.csv"

CSV_DATA = """date,avg_daily_footfall,avg_transaction_value_cad,online_sales_ratio,promo_index,total_daily_sales_cad
2025-03-01,11240,41.2,0.32,0.18,506880
2025-03-02,10810,40.9,0.33,0.15,494120
2025-03-03,11960,41.6,0.31,0.22,529740
2025-03-04,12310,41.8,0.30,0.24,541980
2025-03-05,12680,42.0,0.30,0.26,556440
2025-03-06,12950,42.1,0.31,0.27,566880
2025-03-07,13320,42.3,0.31,0.28,582140
2025-03-08,14010,43.0,0.32,0.30,617820
2025-03-09,13640,42.7,0.33,0.27,603540
2025-03-10,13120,42.2,0.32,0.26,582880
2025-03-11,13490,42.4,0.31,0.28,596260
2025-03-12,13810,42.6,0.31,0.29,610520
2025-03-13,14120,42.8,0.31,0.30,624940
2025-03-14,14560,43.1,0.32,0.32,647880
2025-03-15,15240,43.6,0.33,0.34,689420
2025-03-16,14890,43.4,0.34,0.31,675660
2025-03-17,14280,42.9,0.33,0.30,639720
2025-03-18,14610,43.0,0.32,0.32,653880
2025-03-19,14940,43.2,0.32,0.33,668480
2025-03-20,15210,43.3,0.33,0.34,681940
2025-03-21,15680,43.7,0.33,0.36,713260
2025-03-22,16320,44.2,0.34,0.38,759680
2025-03-23,15940,44.0,0.35,0.35,744920
2025-03-24,15110,43.5,0.34,0.34,692440
2025-03-25,15480,43.6,0.33,0.36,706680
2025-03-26,15810,43.8,0.33,0.37,722060
2025-03-27,16120,44.0,0.34,0.38,740260
2025-03-28,16510,44.3,0.34,0.40,770420
2025-03-29,17180,44.7,0.35,0.42,822980
2025-03-30,16810,44.5,0.36,0.39,807540
2025-03-31,16040,44.1,0.35,0.37,760880
"""


class DatasetWriter:
    """Creates the CSV dataset file if it does not exist."""

    @staticmethod
    def ensure_csv_exists(file_path: str, csv_text: str) -> None:
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)


@dataclass
class RetailRow:
    """Represents one row of retail dataset."""
    date: str
    avg_daily_footfall: float
    avg_transaction_value_cad: float
    online_sales_ratio: float
    promo_index: float
    total_daily_sales_cad: float


class RetailSalesDataset:
    """Reads CSV and converts it into feature matrix X and target vector y."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.rows: list[RetailRow] = []

    def load(self) -> None:
        with open(self.file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(
                    RetailRow(
                        date=r["date"],
                        avg_daily_footfall=float(r["avg_daily_footfall"]),
                        avg_transaction_value_cad=float(r["avg_transaction_value_cad"]),
                        online_sales_ratio=float(r["online_sales_ratio"]),
                        promo_index=float(r["promo_index"]),
                        total_daily_sales_cad=float(r["total_daily_sales_cad"]),
                    )
                )

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        feature_names = [
            "avg_daily_footfall",
            "avg_transaction_value_cad",
            "online_sales_ratio",
            "promo_index",
        ]
        X = np.array(
            [
                [
                    row.avg_daily_footfall,
                    row.avg_transaction_value_cad,
                    row.online_sales_ratio,
                    row.promo_index,
                ]
                for row in self.rows
            ],
            dtype=float,
        )
        y = np.array([row.total_daily_sales_cad for row in self.rows], dtype=float)
        return X, y, feature_names

    def dates(self) -> list[str]:
        return [row.date for row in self.rows]


class RegressionModel(ABC):
    """Abstract base class for regression models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class MultipleLinearRegression(RegressionModel):
    """
    Multiple Linear Regression using Normal Equation:
    beta = (X'X)^(-1) X'y
    Includes intercept via augmentation.
    """

    def __init__(self):
        self.beta: np.ndarray | None = None  # includes intercept as beta[0]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ones = np.ones((X.shape[0], 1), dtype=float)
        X_aug = np.hstack([ones, X])

        # Use pseudo-inverse for numerical stability
        self.beta = np.linalg.pinv(X_aug.T @ X_aug) @ (X_aug.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            raise ValueError("Model is not fitted yet.")
        ones = np.ones((X.shape[0], 1), dtype=float)
        X_aug = np.hstack([ones, X])
        return X_aug @ self.beta


class Metrics:
    """Regression evaluation metrics."""

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))


class PatternRecognizer:
    """
    Basic pattern recognition for tabular regression data:
    - Correlation matrix
    - Residual-based outlier detection (z-score)
    """

    @staticmethod
    def correlation_matrix(X: np.ndarray, feature_names: list[str]) -> None:
        corr = np.corrcoef(X.T)
        print("\nCorrelation Matrix (features):")
        header = " " * 26 + "  ".join([f"{n[:12]:>12}" for n in feature_names])
        print(header)
        for i, name in enumerate(feature_names):
            row_vals = "  ".join([f"{corr[i, j]:12.3f}" for j in range(len(feature_names))])
            print(f"{name:>26}  {row_vals}")

    @staticmethod
    def residual_outliers(dates: list[str], y_true: np.ndarray, y_pred: np.ndarray, z_thresh: float = 2.0) -> None:
        residuals = y_true - y_pred
        std = np.std(residuals)
        if std == 0:
            print("\nOutliers: none (residual std = 0).")
            return
        z = residuals / std
        outlier_idx = np.where(np.abs(z) >= z_thresh)[0]

        print(f"\nResidual Outliers (|z| >= {z_thresh}):")
        if outlier_idx.size == 0:
            print("None detected.")
            return

        for idx in outlier_idx:
            print(
                f"- {dates[idx]} | actual={y_true[idx]:.0f} | predicted={y_pred[idx]:.0f} "
                f"| residual={residuals[idx]:.0f} | z={z[idx]:.2f}"
            )


def main() -> None:
    # 1) Ensure CSV exists (so you can run this script anywhere)
    DatasetWriter.ensure_csv_exists(CSV_FILENAME, CSV_DATA)

    # 2) Load dataset
    dataset = RetailSalesDataset(CSV_FILENAME)
    dataset.load()
    X, y, feature_names = dataset.to_numpy()

    # 3) Fit regression model
    model = MultipleLinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # 4) Print regression results
    r2 = Metrics.r2_score(y, y_pred)
    mse = Metrics.mse(y, y_pred)

    print("=== Regression Results (Multiple Linear Regression) ===")
    print("Intercept (beta0):", round(float(model.beta[0]), 4))
    for i, name in enumerate(feature_names, start=1):
        print(f"Coefficient for {name}: {round(float(model.beta[i]), 6)}")
    print("R^2:", round(float(r2), 4))
    print("MSE:", round(float(mse), 2))

    # 5) Pattern recognition outputs
    PatternRecognizer.correlation_matrix(X, feature_names)
    PatternRecognizer.residual_outliers(dataset.dates(), y, y_pred, z_thresh=2.0)


if __name__ == "__main__":
    main()
