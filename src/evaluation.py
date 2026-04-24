"""
evaluation.py
-------------
Skrip evaluasi model Student Placement yang sudah tersimpan di artifacts/.

Memuat model .pkl dari disk, menjalankan evaluasi pada held-out test set,
mencetak laporan metrik lengkap, dan mencatat hasilnya ke MLflow.

Penggunaan:
    python -m src.evaluation                     # evaluasi kedua model terbaik
    python -m src.evaluation --task clf          # hanya klasifikasi
    python -m src.evaluation --task reg          # hanya regresi
    python -m src.evaluation --model path/to/model.pkl --task clf
"""

import argparse
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)

from src.pre_processing import load_featured
from src.pipeline import FEATURES

ARTIFACTS_DIR = Path('artifacts')
RANDOM_STATE  = 42
TEST_SIZE     = 0.2

MLFLOW_EXPERIMENT_CLF = 'StudentPlacement_Klasifikasi'
MLFLOW_EXPERIMENT_REG = 'StudentPlacement_Regresi'


def _print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _print_metric(name: str, value: float, unit: str = '') -> None:
    print(f"  {name:<20}: {value:.4f} {unit}".rstrip())


def evaluate_classification(model_path: str | Path = None) -> dict:
    """
    Evaluasi model klasifikasi pada test set.

    Parameters
    ----------
    model_path : path ke file .pkl; default = artifacts/best_clf_model.pkl

    Returns
    -------
    dict berisi semua metrik evaluasi
    """
    model_path = Path(model_path) if model_path else ARTIFACTS_DIR / 'best_clf_model.pkl'
    assert model_path.exists(), f"Model tidak ditemukan: {model_path}"

    df = load_featured()
    X  = df[FEATURES]
    y  = (df['placement_status'] == 'Placed').astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    model  = joblib.load(model_path)
    preds  = model.predict(X_test)
    proba  = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':        round(accuracy_score(y_test, preds), 4),
        'precision_macro': round(precision_score(y_test, preds, average='macro'), 4),
        'recall_macro':    round(recall_score(y_test, preds, average='macro'), 4),
        'f1_weighted':     round(f1_score(y_test, preds, average='weighted'), 4),
        'f1_macro':        round(f1_score(y_test, preds, average='macro'), 4),
        'roc_auc':         round(roc_auc_score(y_test, proba), 4),
    }

    _print_section("Evaluasi Klasifikasi")
    print(f"  Model   : {model_path}")
    print(f"  Test set: {len(y_test)} sampel\n")
    for name, val in metrics.items():
        _print_metric(name, val)

    print("\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=['Not Placed', 'Placed']))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Not Placed', 'Actual Placed'],
        columns=['Pred Not Placed', 'Pred Placed'],
    )
    print(cm_df.to_string())

    mlflow.set_experiment(MLFLOW_EXPERIMENT_CLF)
    with mlflow.start_run(run_name='evaluation'):
        mlflow.log_param('model_path', str(model_path))
        mlflow.log_param('n_test', len(y_test))
        mlflow.log_metrics({f'eval_{k}': v for k, v in metrics.items()})

    return metrics


def evaluate_regression(model_path: str | Path = None) -> dict:
    """
    Evaluasi model regresi pada test set (hanya sampel Placed).

    Parameters
    ----------
    model_path : path ke file .pkl; default = artifacts/best_reg_model.pkl

    Returns
    -------
    dict berisi semua metrik evaluasi
    """
    model_path = Path(model_path) if model_path else ARTIFACTS_DIR / 'best_reg_model.pkl'
    assert model_path.exists(), f"Model tidak ditemukan: {model_path}"

    df          = load_featured()
    placed_mask = df['salary_lpa'] > 0
    X_reg       = df.loc[placed_mask, FEATURES]
    y_reg       = df.loc[placed_mask, 'salary_lpa']

    _, X_test, _, y_test = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    model  = joblib.load(model_path)
    preds  = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    metrics = {
        'rmse': round(rmse, 4),
        'mae':  round(float(mean_absolute_error(y_test, preds)), 4),
        'r2':   round(float(r2_score(y_test, preds)), 4),
    }

    _print_section("Evaluasi Regresi")
    print(f"  Model   : {model_path}")
    print(f"  Test set: {len(y_test)} sampel (Placed only)\n")
    _print_metric('RMSE', metrics['rmse'], 'LPA')
    _print_metric('MAE',  metrics['mae'],  'LPA')
    _print_metric('R2',   metrics['r2'])

    r2 = metrics['r2']
    if r2 >= 0.7:
        interp = "Baik - model menjelaskan lebih dari 70% variasi gaji"
    elif r2 >= 0.4:
        interp = "Cukup - ada pola yang tertangkap, namun masih terbatas"
    else:
        interp = "Lemah - banyak variasi gaji yang tidak tertangkap fitur ini"
    print(f"\n  Interpretasi R2 : {interp}")

    residuals = y_test.values - preds
    print(f"\n  Statistik Residual:")
    print(f"    Mean   : {residuals.mean():.4f} LPA")
    print(f"    Std    : {residuals.std():.4f} LPA")
    print(f"    Min    : {residuals.min():.4f} LPA")
    print(f"    Max    : {residuals.max():.4f} LPA")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_REG)
    with mlflow.start_run(run_name='evaluation'):
        mlflow.log_param('model_path', str(model_path))
        mlflow.log_param('n_test', len(y_test))
        mlflow.log_metrics({f'eval_{k}': v for k, v in metrics.items()})

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluasi model Student Placement dari artifacts/'
    )
    parser.add_argument(
        '--task',
        choices=['clf', 'reg', 'all'],
        default='all',
        help='Tugas yang dievaluasi',
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Path ke file .pkl spesifik (opsional)',
    )
    args = parser.parse_args()

    if args.task in ('clf', 'all'):
        evaluate_classification(args.model)

    if args.task in ('reg', 'all'):
        evaluate_regression(args.model if args.task == 'reg' else None)


if __name__ == '__main__':
    main()
