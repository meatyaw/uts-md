"""
run_experiment.py
-----------------
Skrip orkestrator utama yang menjalankan seluruh alur eksperimen
dari awal hingga akhir dalam satu perintah.

Urutan eksekusi:
  1. Data Ingestion   - validasi dan merge dataset
  2. Feature Engineering (via load_featured di dalam train)
  3. Training         - semua model klasifikasi dan regresi
  4. Evaluation       - laporan metrik model terbaik
  5. Ringkasan akhir  - tabel perbandingan semua model

Penggunaan:
    python run_experiment.py
    python run_experiment.py --task clf
    python run_experiment.py --task reg
    python run_experiment.py --task all --verbose
"""

import argparse
import time
from pathlib import Path

from src.data_ingestion import ingest
from src.train import train_classification, train_regression
from src.evaluation import evaluate_classification, evaluate_regression

def print_header(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def print_step(step: int, description: str) -> None:
    print(f"\n[Step {step}] {description}")
    print("-" * 40)


def print_comparison_table(results: dict, metric_order: list) -> None:
    """Cetak tabel perbandingan model dalam format teks."""
    if not results:
        return
    col_w = 22
    metric_w = 12

    # Header
    header = f"{'Model':<{col_w}}" + "".join(f"{m:>{metric_w}}" for m in metric_order)
    print(header)
    print("-" * (col_w + metric_w * len(metric_order)))

    # Baris data
    for model_name, metrics in results.items():
        row = f"{model_name:<{col_w}}"
        for m in metric_order:
            val = metrics.get(m, float('nan'))
            row += f"{val:>{metric_w}.4f}"
        print(row)


def run(task: str = 'all', verbose: bool = False) -> None:
    start_time = time.time()

    print_header("Student Placement - Full Experiment Pipeline")
    print(f"  Task    : {task}")
    print(f"  Verbose : {verbose}")

    print_step(1, "Data Ingestion & Validasi")
    df = ingest(
        features_path='data/raw/A.csv',
        targets_path='data/raw/A_targets.csv',
        save_merged=True,
    )
    print(f"Dataset siap: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

    clf_results = {}
    reg_results = {}
    step = 2

    if task in ('clf', 'all'):
        print_step(step, "Training - Klasifikasi (placement_status)")
        clf_results = train_classification()
        step += 1

    if task in ('reg', 'all'):
        print_step(step, "Training - Regresi (salary_lpa)")
        reg_results = train_regression()
        step += 1

    if task in ('clf', 'all'):
        print_step(step, "Evaluasi - Model Klasifikasi Terbaik")
        clf_eval = evaluate_classification()
        step += 1

    if task in ('reg', 'all'):
        print_step(step, "Evaluasi - Model Regresi Terbaik")
        reg_eval = evaluate_regression()
        step += 1
    print_header("Ringkasan Akhir - Perbandingan Semua Model")

    if clf_results:
        print("\nKlasifikasi (placement_status):")
        print_comparison_table(
            clf_results,
            metric_order=['cv_f1_weighted', 'test_f1_weighted', 'test_roc_auc', 'test_accuracy'],
        )
        best_clf = max(clf_results, key=lambda n: clf_results[n]['test_roc_auc'])
        print(f"\nModel klasifikasi terbaik : {best_clf}")
        print(f"  ROC-AUC : {clf_results[best_clf]['test_roc_auc']:.4f}")
        print(f"  F1-W    : {clf_results[best_clf]['test_f1_weighted']:.4f}")

    if reg_results:
        print("\nRegresi (salary_lpa - Placed only):")
        print_comparison_table(
            reg_results,
            metric_order=['cv_rmse', 'test_rmse', 'test_mae', 'test_r2'],
        )
        best_reg = max(reg_results, key=lambda n: reg_results[n]['test_r2'])
        print(f"\nModel regresi terbaik : {best_reg}")
        print(f"  R2   : {reg_results[best_reg]['test_r2']:.4f}")
        print(f"  RMSE : {reg_results[best_reg]['test_rmse']:.4f} LPA")

    elapsed = time.time() - start_time
    print_header(f"Eksperimen selesai dalam {elapsed:.1f} detik")
    print("Artifacts tersimpan di  : artifacts/")
    print("MLflow tracking         : mlruns/")
    print("Lihat UI MLflow dengan  : mlflow ui")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Jalankan full experiment pipeline Student Placement'
    )
    parser.add_argument(
        '--task',
        choices=['clf', 'reg', 'all'],
        default='all',
        help='Tugas yang dijalankan (default: all)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Tampilkan output lebih detail',
    )
    args = parser.parse_args()
    run(task=args.task, verbose=args.verbose)
