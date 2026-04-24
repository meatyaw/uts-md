"""
pipeline.py
-----------
Definisi fitur dan fungsi factory untuk membangun objek Pipeline sklearn.

Setiap pipeline mengintegrasikan:
  preprocessing (imputation + encoding/scaling) + model

dalam satu objek sehingga tidak ada data leakage: statistik transformasi
(mean, std, encoder mapping) hanya dihitung dari data training.

Tersedia:
- get_clf_pipeline(name)  : pipeline untuk tugas klasifikasi
- get_reg_pipeline(name)  : pipeline untuk tugas regresi
- CLF_CONFIGS / REG_CONFIGS : dict konfigurasi default hyperparameter
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Definisi kolom fitur
# ---------------------------------------------------------------------------
CAT_FEATURES = [
    'gender',
    'branch',
    'part_time_job',
    'family_income_level',
    'city_tier',
    'internet_access',
    'extracurricular_involvement',
]

NUM_FEATURES = [
    'cgpa',
    'backlogs',
    'study_hours_per_day',
    'attendance_percentage',
    'projects_completed',
    'internships_completed',
    'coding_skill_rating',
    'communication_skill_rating',
    'aptitude_skill_rating',
    'hackathons_participated',
    'certifications_count',
    'sleep_hours',
    'stress_level',
    # fitur hasil feature engineering
    'academic_avg',
    'total_skill_score',
    'experience_score',
    'high_risk_backlog',
]

FEATURES = CAT_FEATURES + NUM_FEATURES

# ---------------------------------------------------------------------------
# Konfigurasi hyperparameter default untuk setiap model
# ---------------------------------------------------------------------------
CLF_CONFIGS = {
    'logistic_regression': {
        'model': LogisticRegression,
        'params': {
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42,
        },
    },
    'random_forest': {
        'model': RandomForestClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 10,
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': 42,
        },
    },
    'xgboost': {
        'model': XGBClassifier,
        'params': {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': 42,
        },
    },
    'lightgbm': {
        'model': LGBMClassifier,
        'params': {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'class_weight': 'balanced',
            'verbose': -1,
            'random_state': 42,
        },
    },
    'gradient_boosting': {
        'model': GradientBoostingClassifier,
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.08,
            'max_depth': 5,
            'random_state': 42,
        },
    },
}

REG_CONFIGS = {
    'ridge': {
        'model': Ridge,
        'params': {'alpha': 1.0},
    },
    'random_forest': {
        'model': RandomForestRegressor,
        'params': {
            'n_estimators': 200,
            'max_depth': 12,
            'n_jobs': -1,
            'random_state': 42,
        },
    },
    'xgboost': {
        'model': XGBRegressor,
        'params': {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'verbosity': 0,
            'random_state': 42,
        },
    },
    'lightgbm': {
        'model': LGBMRegressor,
        'params': {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'verbose': -1,
            'random_state': 42,
        },
    },
    'gradient_boosting': {
        'model': GradientBoostingRegressor,
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.08,
            'max_depth': 5,
            'random_state': 42,
        },
    },
}


# ---------------------------------------------------------------------------
# Fungsi factory pembuatan preprocessor
# ---------------------------------------------------------------------------
def _build_preprocessor() -> ColumnTransformer:
    """
    Buat ColumnTransformer untuk fitur kategorik dan numerik.

    Kategorik : imputasi dengan konstanta 'Unknown' -> OrdinalEncoder
                (missing dijadikan kategori tersendiri agar informasi terjaga)
    Numerik   : imputasi dengan median (lebih robust terhadap outlier) -> StandardScaler
    """
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )),
    ])

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])

    return ColumnTransformer([
        ('cat', cat_transformer, CAT_FEATURES),
        ('num', num_transformer, NUM_FEATURES),
    ])


# ---------------------------------------------------------------------------
# Fungsi factory pembuatan pipeline
# ---------------------------------------------------------------------------
def get_clf_pipeline(model_name: str = 'logistic_regression') -> Pipeline:
    """
    Kembalikan Pipeline sklearn untuk klasifikasi.

    Parameters
    ----------
    model_name : salah satu dari CLF_CONFIGS.keys()
                 ('logistic_regression', 'random_forest', 'xgboost',
                  'lightgbm', 'gradient_boosting')

    Returns
    -------
    sklearn.pipeline.Pipeline siap di-fit
    """
    if model_name not in CLF_CONFIGS:
        raise ValueError(
            f"Model '{model_name}' tidak dikenal. "
            f"Pilihan: {list(CLF_CONFIGS.keys())}"
        )
    cfg = CLF_CONFIGS[model_name]
    return Pipeline([
        ('preprocessor', _build_preprocessor()),
        ('classifier',   cfg['model'](**cfg['params'])),
    ])


def get_reg_pipeline(model_name: str = 'ridge') -> Pipeline:
    """
    Kembalikan Pipeline sklearn untuk regresi.

    Parameters
    ----------
    model_name : salah satu dari REG_CONFIGS.keys()
                 ('ridge', 'random_forest', 'xgboost',
                  'lightgbm', 'gradient_boosting')

    Returns
    -------
    sklearn.pipeline.Pipeline siap di-fit
    """
    if model_name not in REG_CONFIGS:
        raise ValueError(
            f"Model '{model_name}' tidak dikenal. "
            f"Pilihan: {list(REG_CONFIGS.keys())}"
        )
    cfg = REG_CONFIGS[model_name]
    return Pipeline([
        ('preprocessor', _build_preprocessor()),
        ('regressor',    cfg['model'](**cfg['params'])),
    ])


if __name__ == '__main__':
    print("Klasifikasi :", list(CLF_CONFIGS.keys()))
    print("Regresi     :", list(REG_CONFIGS.keys()))
    print("Fitur total :", len(FEATURES))
    pipe = get_clf_pipeline('xgboost')
    print("Pipeline XGB:", pipe)
