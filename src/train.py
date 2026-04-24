
import argparse
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

from src.pre_processing import load_featured
from src.pipeline import (
    get_clf_pipeline, get_reg_pipeline,
    CLF_CONFIGS, REG_CONFIGS, FEATURES,
)

ARTIFACTS_DIR    = Path('artifacts')
RANDOM_STATE     = 42
TEST_SIZE        = 0.2
CV_FOLDS         = 5
MLFLOW_EXPERIMENT_CLF = 'StudentPlacement_Klasifikasi'
MLFLOW_EXPERIMENT_REG = 'StudentPlacement_Regresi'


def _log_params(model_name: str, config: dict, dataset_info: dict) -> None:
    """Log hyperparameter model dan informasi dataset ke MLflow."""
    mlflow.log_param('model_name', model_name)
    mlflow.log_param('test_size', TEST_SIZE)
    mlflow.log_param('cv_folds', CV_FOLDS)
    mlflow.log_param('n_features', len(FEATURES))
    for k, v in dataset_info.items():
        mlflow.log_param(k, v)
    for k, v in config['params'].items():
        mlflow.log_param(f'model_{k}', v)


def train_classification(model_name: str = None) -> dict:
    
    df = load_featured()
    X  = df[FEATURES]
    y  = (df['placement_status'] == 'Placed').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    dataset_info = {
        'n_train':        len(X_train),
        'n_test':         len(X_test),
        'class_balance':  round(y_train.mean(), 4),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models_to_run = (
        {model_name: CLF_CONFIGS[model_name]}
        if model_name
        else CLF_CONFIGS
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_CLF)
    all_results = {}
    best_auc   = -1.0
    best_pipe  = None
    best_name  = None

    for name, config in models_to_run.items():
        print(f"\n[CLF] Training: {name}")
        with mlflow.start_run(run_name=name):
            pipe = get_clf_pipeline(name)

            # Cross-validation di training set
            cv_f1 = cross_val_score(
                pipe, X_train, y_train, cv=cv, scoring='f1_weighted',
            ).mean()

            # Fit full training set
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            # Hitung metrik
            metrics = {
                'cv_f1_weighted': round(cv_f1, 4),
                'test_accuracy':  round(accuracy_score(y_test, y_pred), 4),
                'test_f1_weighted': round(f1_score(y_test, y_pred, average='weighted'), 4),
                'test_roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
            }

            # Log ke MLflow
            _log_params(name, config, dataset_info)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path='model')

            all_results[name] = metrics
            print(f"  CV F1-W       : {metrics['cv_f1_weighted']:.4f}")
            print(f"  Test Accuracy : {metrics['test_accuracy']:.4f}")
            print(f"  Test F1-W     : {metrics['test_f1_weighted']:.4f}")
            print(f"  Test ROC-AUC  : {metrics['test_roc_auc']:.4f}")

            if metrics['test_roc_auc'] > best_auc:
                best_auc  = metrics['test_roc_auc']
                best_pipe = pipe
                best_name = name

    if best_pipe is not None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ARTIFACTS_DIR / 'best_clf_model.pkl'
        joblib.dump(best_pipe, out_path)
        print(f"\nModel klasifikasi terbaik : {best_name} (AUC={best_auc:.4f})")
        print(f"Disimpan ke               : {out_path}")

    return all_results


def train_regression(model_name: str = None) -> dict:
   
    df          = load_featured()
    placed_mask = df['salary_lpa'] > 0
    X_reg       = df.loc[placed_mask, FEATURES]
    y_reg       = df.loc[placed_mask, 'salary_lpa']

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    dataset_info = {
        'n_train':     len(X_train),
        'n_test':      len(X_test),
        'salary_mean': round(y_train.mean(), 4),
        'salary_std':  round(y_train.std(), 4),
    }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models_to_run = (
        {model_name: REG_CONFIGS[model_name]}
        if model_name
        else REG_CONFIGS
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_REG)
    all_results = {}
    best_r2    = -np.inf
    best_pipe  = None
    best_name  = None

    for name, config in models_to_run.items():
        print(f"\n[REG] Training: {name}")
        with mlflow.start_run(run_name=name):
            pipe = get_reg_pipeline(name)

            # Cross-validation di training set
            cv_rmse = -cross_val_score(
                pipe, X_train, y_train, cv=cv,
                scoring='neg_root_mean_squared_error',
            ).mean()

            # Fit full training set
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            # Hitung metrik
            metrics = {
                'cv_rmse':   round(cv_rmse, 4),
                'test_rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                'test_mae':  round(mean_absolute_error(y_test, y_pred), 4),
                'test_r2':   round(r2_score(y_test, y_pred), 4),
            }

            _log_params(name, config, dataset_info)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path='model')

            all_results[name] = metrics
            print(f"  CV RMSE  : {metrics['cv_rmse']:.4f}")
            print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
            print(f"  Test MAE : {metrics['test_mae']:.4f}")
            print(f"  Test R2  : {metrics['test_r2']:.4f}")

            if metrics['test_r2'] > best_r2:
                best_r2   = metrics['test_r2']
                best_pipe = pipe
                best_name = name

    if best_pipe is not None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ARTIFACTS_DIR / 'best_reg_model.pkl'
        joblib.dump(best_pipe, out_path)
        print(f"\nModel regresi terbaik : {best_name} (R2={best_r2:.4f})")
        print(f"Disimpan ke           : {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Train Student Placement models dengan MLflow tracking'
    )
    parser.add_argument(
        '--task',
        choices=['clf', 'reg', 'all'],
        default='all',
        help='Tugas yang dijalankan: clf=klasifikasi, reg=regresi, all=keduanya',
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Nama model spesifik (opsional). Contoh: xgboost, lightgbm, ridge',
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Student Placement - Training Pipeline")
    print("=" * 60)

    if args.task in ('clf', 'all'):
        print("\nMulai training KLASIFIKASI...")
        clf_results = train_classification(args.model)

    if args.task in ('reg', 'all'):
        print("\nMulai training REGRESI...")
        reg_results = train_regression(args.model)

    print("\n" + "=" * 60)
    print("Training selesai.")
    print("Jalankan 'mlflow ui' untuk melihat hasil eksperimen.")
    print("=" * 60)


if __name__ == '__main__':
    main()
