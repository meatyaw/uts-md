"""
pre_processing.py
-----------------
Modul untuk feature engineering pada dataset Student Placement.

Setiap transformasi di sini bersifat deterministik dan tidak bergantung
pada statistik training set, sehingga aman dilakukan sebelum train-test split.
Transformasi yang bergantung pada statistik (scaling, imputation) dikerjakan
di dalam pipeline sklearn agar tidak terjadi data leakage.

Fungsi utama:
- engineer_features() : tambahkan fitur turunan ke DataFrame
- load_featured()     : gabungkan ingest + feature engineering dalam satu langkah
"""

import pandas as pd
from pathlib import Path

from src.data_ingestion import ingest


# ---------------------------------------------------------------------------
# Fungsi utama feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan fitur-fitur turunan ke DataFrame.

    Fitur yang dibuat:
    - cgpa_norm       : CGPA dinormalisasi ke skala 0-100 agar sebanding
                        dengan tenth_percentage dan twelfth_percentage
    - academic_avg    : rata-rata dari tiga metrik akademik (tenth, twelfth, cgpa_norm)
                        untuk mereduksi multikolinearitas
    - total_skill_score : jumlah rating coding + communication + aptitude
    - experience_score  : skor pengalaman berbobot (magang x2 lebih bernilai
                          dibanding proyek atau hackathon di mata industri)
    - high_risk_backlog : flag biner; HR memberi sinyal merah sama untuk
                          1 atau 4 backlog, sehingga relasi tidak linier

    Parameters
    ----------
    df : DataFrame hasil merge (output dari ingest atau load_raw)

    Returns
    -------
    DataFrame baru dengan kolom-kolom fitur tambahan
    """
    df = df.copy()

    # 1. Normalisasi CGPA ke skala persentase
    df['cgpa_norm'] = df['cgpa'] * 10

    # 2. Rata-rata akademik lintas tiga jenjang
    df['academic_avg'] = (
        df['tenth_percentage']
        + df['twelfth_percentage']
        + df['cgpa_norm']
    ) / 3

    # 3. Total kemampuan teknis dan kognitif
    df['total_skill_score'] = (
        df['coding_skill_rating']
        + df['communication_skill_rating']
        + df['aptitude_skill_rating']
    )

    # 4. Skor pengalaman industri berbobot
    df['experience_score'] = (
        df['internships_completed'] * 2
        + df['projects_completed']
        + df['hackathons_participated']
    )

    # 5. Flag risiko backlog (biner)
    df['high_risk_backlog'] = (df['backlogs'] > 0).astype(int)

    return df


def load_featured(
    features_path: str | Path = 'data/raw/A.csv',
    targets_path:  str | Path = 'data/raw/A_targets.csv',
) -> pd.DataFrame:
    """
    Muat data mentah + terapkan feature engineering dalam satu langkah.

    Returns
    -------
    pd.DataFrame siap pakai untuk modeling
    """
    df_raw = ingest(
        features_path=features_path,
        targets_path=targets_path,
        save_merged=False,
    )
    return engineer_features(df_raw)


if __name__ == '__main__':
    df = load_featured()
    new_cols = ['cgpa_norm', 'academic_avg', 'total_skill_score',
                'experience_score', 'high_risk_backlog']
    print(f"Shape setelah feature engineering: {df.shape}")
    print(df[new_cols].describe().T.to_string())
