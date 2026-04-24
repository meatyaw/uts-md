import pandas as pd
from pathlib import Path

RAW_DIR         = Path('data/raw')
FEATURES_FILE   = RAW_DIR / 'A.csv'
TARGETS_FILE    = RAW_DIR / 'A_targets.csv'
MERGED_FILE     = RAW_DIR / 'merged.csv'

REQUIRED_FEATURE_COLS = {
    'Student_ID', 'gender', 'branch', 'cgpa',
    'tenth_percentage', 'twelfth_percentage', 'backlogs',
    'study_hours_per_day', 'attendance_percentage',
    'projects_completed', 'internships_completed',
    'coding_skill_rating', 'communication_skill_rating',
    'aptitude_skill_rating', 'hackathons_participated',
    'certifications_count', 'sleep_hours', 'stress_level',
    'part_time_job', 'family_income_level', 'city_tier',
    'internet_access', 'extracurricular_involvement',
}

REQUIRED_TARGET_COLS = {
    'Student_ID', 'placement_status', 'salary_lpa',
}


def _validate_columns(df: pd.DataFrame, required: set, filename: str) -> None:
    """Periksa apakah semua kolom yang dibutuhkan tersedia."""
    missing = required - set(df.columns)
    assert not missing, (
        f"[{filename}] Kolom tidak ditemukan: {sorted(missing)}"
    )


def _validate_not_empty(df: pd.DataFrame, filename: str) -> None:
    """Pastikan DataFrame tidak kosong."""
    assert not df.empty, f"[{filename}] Dataset kosong."


def ingest(
    features_path: str | Path = FEATURES_FILE,
    targets_path:  str | Path = TARGETS_FILE,
    save_merged:   bool = True,
) -> pd.DataFrame:
    """
    Muat dan validasi kedua file CSV, lalu gabungkan menjadi satu DataFrame.

    Parameters
    ----------
    features_path : path ke A.csv
    targets_path  : path ke A_targets.csv
    save_merged   : jika True, simpan hasil merge ke data/raw/merged.csv

    Returns
    -------
    pd.DataFrame gabungan fitur + target, tanpa kolom Student_ID
    """
    df_features = pd.read_csv(features_path)
    df_targets  = pd.read_csv(targets_path)

    _validate_columns(df_features, REQUIRED_FEATURE_COLS, features_path)
    _validate_columns(df_targets,  REQUIRED_TARGET_COLS,  targets_path)
    _validate_not_empty(df_features, features_path)
    _validate_not_empty(df_targets,  targets_path)

    assert len(df_features) == len(df_targets), (
        f"Jumlah baris tidak cocok: features={len(df_features)}, "
        f"targets={len(df_targets)}"
    )

    df = pd.merge(df_features, df_targets, on='Student_ID', how='inner')
    df.drop(columns=['Student_ID'], inplace=True)

    if save_merged:
        MERGED_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(MERGED_FILE, index=False)

    print(f"Ingested  : {len(df_features)} rows dari {features_path}")
    print(f"Merged    : {df.shape[0]} rows x {df.shape[1]} cols -> {MERGED_FILE if save_merged else 'tidak disimpan'}")
    return df


def load_raw(
    features_path: str | Path = FEATURES_FILE,
    targets_path:  str | Path = TARGETS_FILE,
) -> pd.DataFrame:
    
    df_features = pd.read_csv(features_path)
    df_targets  = pd.read_csv(targets_path)
    df = pd.merge(df_features, df_targets, on='Student_ID', how='inner')
    df.drop(columns=['Student_ID'], inplace=True)
    return df


if __name__ == '__main__':
    df = ingest()
    print(df.dtypes)
    print(df.head(3))
