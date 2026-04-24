import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="UTS MD",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background utama */
.stApp {
    background: #0f1117;
    color: #e8e8e8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #2a2f3d;
}

[data-testid="stSidebar"] * {
    color: #c8cdd8 !important;
}

/* Judul utama */
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f5c842;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.sub-title {
    font-size: 0.95rem;
    color: #7a8099;
    margin-bottom: 2rem;
    font-family: 'Space Mono', monospace;
}

/* Kartu metrik */
.metric-card {
    background: #161b27;
    border: 1px solid #2a2f3d;
    border-left: 3px solid #f5c842;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

.metric-card h4 {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #7a8099;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f5c842;
}

.metric-card .sub {
    font-size: 0.8rem;
    color: #7a8099;
    margin-top: 0.2rem;
}

/* Result banner */
.result-placed {
    background: linear-gradient(135deg, #0d2b1f, #133d2b);
    border: 1px solid #1e6b47;
    border-left: 4px solid #3ecf8e;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    text-align: center;
}

.result-not-placed {
    background: linear-gradient(135deg, #2b0d0d, #3d1313);
    border: 1px solid #6b1e1e;
    border-left: 4px solid #cf3e3e;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    text-align: center;
}

.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

/* Section header */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #f5c842;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #2a2f3d;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Input labels */
.stSelectbox label, .stSlider label, .stNumberInput label {
    font-size: 0.82rem !important;
    color: #9ba3b8 !important;
    font-weight: 500 !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    background: #f5c842 !important;
    color: #0f1117 !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 2rem !important;
    border-radius: 6px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #7a8099 !important;
}

.stTabs [aria-selected="true"] {
    color: #f5c842 !important;
    border-bottom-color: #f5c842 !important;
}

/* Divider */
hr {
    border-color: #2a2f3d;
}

/* Info box */
.info-box {
    background: #161b27;
    border: 1px solid #2a2f3d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #9ba3b8;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

ARTIFACTS_DIR = Path('artifacts')
CLF_MODEL_PATH = ARTIFACTS_DIR / 'best_clf_model.pkl'
REG_MODEL_PATH = ARTIFACTS_DIR / 'best_reg_model.pkl'

CAT_FEATURES = [
    'gender', 'branch', 'part_time_job', 'family_income_level',
    'city_tier', 'internet_access', 'extracurricular_involvement',
]
NUM_FEATURES = [
    'cgpa', 'backlogs', 'study_hours_per_day', 'attendance_percentage',
    'projects_completed', 'internships_completed', 'coding_skill_rating',
    'communication_skill_rating', 'aptitude_skill_rating',
    'hackathons_participated', 'certifications_count', 'sleep_hours',
    'stress_level', 'academic_avg', 'total_skill_score',
    'experience_score', 'high_risk_backlog',
]
FEATURES = CAT_FEATURES + NUM_FEATURES


@st.cache_resource
def load_models():
    clf_model, reg_model = None, None
    if CLF_MODEL_PATH.exists():
        clf_model = joblib.load(CLF_MODEL_PATH)
    if REG_MODEL_PATH.exists():
        reg_model = joblib.load(REG_MODEL_PATH)
    return clf_model, reg_model


def engineer_features(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df['cgpa_norm']         = df['cgpa'] * 10
    df['academic_avg']      = (df['tenth_percentage'] + df['twelfth_percentage'] + df['cgpa_norm']) / 3
    df['total_skill_score'] = df['coding_skill_rating'] + df['communication_skill_rating'] + df['aptitude_skill_rating']
    df['experience_score']  = df['internships_completed'] * 2 + df['projects_completed'] + df['hackathons_participated']
    df['high_risk_backlog'] = (df['backlogs'] > 0).astype(int)
    return df[FEATURES]

def make_gauge(prob: float) -> go.Figure:
    color = "#3ecf8e" if prob >= 0.5 else "#cf3e3e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%", 'font': {'size': 36, 'color': color, 'family': 'Space Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#7a8099', 'tickfont': {'color': '#7a8099', 'size': 10}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#161b27',
            'bordercolor': '#2a2f3d',
            'steps': [
                {'range': [0, 50], 'color': '#1a0d0d'},
                {'range': [50, 100], 'color': '#0d1a12'},
            ],
            'threshold': {
                'line': {'color': '#f5c842', 'width': 2},
                'thickness': 0.8,
                'value': 50,
            }
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e8e8e8'},
    )
    return fig

def make_radar(row: dict) -> go.Figure:
    categories = ['Academic', 'Coding', 'Communication', 'Aptitude', 'Experience', 'Attendance']
    values = [
        min(row.get('cgpa', 0) * 10, 100),
        row.get('coding_skill_rating', 0) * 10,
        row.get('communication_skill_rating', 0) * 10,
        row.get('aptitude_skill_rating', 0) * 10,
        min((row.get('internships_completed', 0) * 2 + row.get('projects_completed', 0)) * 15, 100),
        row.get('attendance_percentage', 0),
    ]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(245,200,66,0.12)',
        line=dict(color='#f5c842', width=2),
        marker=dict(color='#f5c842', size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#2a2f3d', tickfont={'color': '#7a8099', 'size': 9}, color='#7a8099'),
            angularaxis=dict(gridcolor='#2a2f3d', tickfont={'color': '#c8cdd8', 'size': 10}),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        showlegend=False,
    )
    return fig


with st.sidebar:
    st.markdown("### Student Data Input")
    st.markdown("---")

    st.markdown('<div class="section-header">Identitas</div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    branch = st.selectbox("Branch / Jurusan", [
        "Computer Science", "Electronics", "Mechanical", "Civil",
        "Information Technology", "Electrical", "Other"
    ])
    city_tier   = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    part_time_job   = st.selectbox("Part Time Job", ["Yes", "No"])
    extracurricular = st.selectbox("Extracurricular Involvement", ["Low", "Medium", "High"])

    st.markdown('<div class="section-header">Akademik</div>', unsafe_allow_html=True)
    tenth_pct  = st.slider("Nilai SMA (10th %)", 40.0, 100.0, 75.0, step=0.5)
    twelfth_pct = st.slider("Nilai SMK/SMA (12th %)", 40.0, 100.0, 72.0, step=0.5)
    cgpa       = st.slider("CGPA (0–10)", 4.0, 10.0, 7.5, step=0.1)
    backlogs   = st.number_input("Jumlah Backlog", 0, 20, 0)
    attendance = st.slider("Attendance (%)", 50.0, 100.0, 85.0, step=0.5)

    st.markdown('<div class="section-header">Kemampuan (1–10)</div>', unsafe_allow_html=True)
    coding_skill = st.slider("Coding Skill", 1, 10, 6)
    comm_skill   = st.slider("Communication Skill", 1, 10, 7)
    apt_skill    = st.slider("Aptitude Skill", 1, 10, 6)

    st.markdown('<div class="section-header">Pengalaman & Kebiasaan</div>', unsafe_allow_html=True)
    internships  = st.number_input("Internships Completed", 0, 10, 1)
    projects     = st.number_input("Projects Completed", 0, 20, 2)
    hackathons   = st.number_input("Hackathons Participated", 0, 20, 1)
    certifications = st.number_input("Certifications Count", 0, 30, 2)
    study_hours  = st.slider("Study Hours/Day", 0.0, 16.0, 4.0, step=0.5)
    sleep_hours  = st.slider("Sleep Hours/Day", 3.0, 12.0, 7.0, step=0.5)
    stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)

    st.markdown("---")
    predict_btn = st.button("PREDICT NOW")


st.markdown('<div class="main-title">Student Placement<br>Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">// Powered by ML · Classification + Salary Regression</div>', unsafe_allow_html=True)

clf_model, reg_model = load_models()

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    status_clf = "Loaded" if clf_model else " Not Found"
    st.markdown(f'<div class="metric-card"><h4>CLF MODEL</h4><div class="value" style="font-size:1rem">{status_clf}</div><div class="sub">best_clf_model.pkl</div></div>', unsafe_allow_html=True)
with col_s2:
    status_reg = "Loaded" if reg_model else " Not Found"
    st.markdown(f'<div class="metric-card"><h4>REG MODEL</h4><div class="value" style="font-size:1rem">{status_reg}</div><div class="sub">best_reg_model.pkl</div></div>', unsafe_allow_html=True)
with col_s3:
    st.markdown(f'<div class="metric-card"><h4>FEATURES</h4><div class="value" style="font-size:1rem">24</div><div class="sub">input + engineered</div></div>', unsafe_allow_html=True)

st.markdown("---")

input_row = {
    'gender': gender,
    'branch': branch,
    'part_time_job': part_time_job,
    'family_income_level': family_income,
    'city_tier': city_tier,
    'internet_access': internet_access,
    'extracurricular_involvement': extracurricular,
    'cgpa': cgpa,
    'backlogs': backlogs,
    'study_hours_per_day': study_hours,
    'attendance_percentage': attendance,
    'projects_completed': projects,
    'internships_completed': internships,
    'coding_skill_rating': coding_skill,
    'communication_skill_rating': comm_skill,
    'aptitude_skill_rating': apt_skill,
    'hackathons_participated': hackathons,
    'certifications_count': certifications,
    'sleep_hours': sleep_hours,
    'stress_level': stress_level,
    'tenth_percentage': tenth_pct,
    'twelfth_percentage': twelfth_pct,
}

tab1, tab2, tab3 = st.tabs(["  Prediction Result", "  Student Profile", " How It Works"])

with tab1:
    if not predict_btn:
        st.markdown('<div class="info-box"> Isi data mahasiswa di sidebar kiri, lalu klik <b>PREDICT NOW</b> untuk melihat hasil prediksi.</div>', unsafe_allow_html=True)
    else:
        if clf_model is None and reg_model is None:
            st.error(" Tidak ada model yang ditemukan di `artifacts/`. Jalankan `python run_experiment.py` terlebih dahulu.")
        else:
            X_input = engineer_features(input_row)

            if clf_model:
                pred_class = clf_model.predict(X_input)[0]
                pred_proba = clf_model.predict_proba(X_input)[0]
                prob_placed = pred_proba[1]

                is_placed = pred_class == 1

                if is_placed:
                    st.markdown(f"""
                    <div class="result-placed">
                        <div class="result-title" style="color:#3ecf8e"> PLACED</div>
                        <div style="color:#a8e6cf;font-size:0.9rem">Mahasiswa ini diprediksi berhasil mendapat pekerjaan.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-not-placed">
                        <div class="result-title" style="color:#cf3e3e"> NOT PLACED</div>
                        <div style="color:#e6a8a8;font-size:0.9rem">Mahasiswa ini diprediksi belum berhasil mendapat pekerjaan.</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                col_g, col_m = st.columns([1.2, 1])
                with col_g:
                    st.markdown('<div class="section-header">Placement Probability</div>', unsafe_allow_html=True)
                    st.plotly_chart(make_gauge(prob_placed), use_container_width=True)
                with col_m:
                    st.markdown('<div class="section-header">Probabilitas Detail</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><h4>PROB PLACED</h4><div class="value">{prob_placed*100:.1f}%</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><h4>PROB NOT PLACED</h4><div class="value" style="color:#cf3e3e">{(1-prob_placed)*100:.1f}%</div></div>', unsafe_allow_html=True)

                    conf_label = "Tinggi" if max(prob_placed, 1-prob_placed) > 0.8 else ("Sedang" if max(prob_placed, 1-prob_placed) > 0.6 else "Rendah")
                    st.markdown(f'<div class="metric-card"><h4>CONFIDENCE</h4><div class="value" style="font-size:1.1rem">{conf_label}</div></div>', unsafe_allow_html=True)

            
            st.markdown("---")
            st.markdown('<div class="section-header">Estimasi Gaji (Placed Only)</div>', unsafe_allow_html=True)

            if reg_model and (not clf_model or is_placed):
                salary_pred = reg_model.predict(X_input)[0]
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.markdown(f'<div class="metric-card"><h4>ESTIMASI GAJI</h4><div class="value">₹{salary_pred:.2f}</div><div class="sub">LPA (Lakhs Per Annum)</div></div>', unsafe_allow_html=True)
                with col_r2:
                    monthly = (salary_pred * 100000) / 12
                    st.markdown(f'<div class="metric-card"><h4>PER BULAN</h4><div class="value" style="font-size:1.3rem">₹{monthly:,.0f}</div><div class="sub">estimasi bulanan</div></div>', unsafe_allow_html=True)
                with col_r3:
                    tier = "Senior" if salary_pred >= 8 else ("Mid-level" if salary_pred >= 5 else "Entry-level")
                    st.markdown(f'<div class="metric-card"><h4>SALARY TIER</h4><div class="value" style="font-size:1.1rem">{tier}</div></div>', unsafe_allow_html=True)

                # Visualisasi gaji vs rata-rata pasar (simulasi)
                salary_data = pd.DataFrame({
                    'Kategori': ['Entry (3 LPA)', 'Your Prediction', 'Mid (6 LPA)', 'Senior (10 LPA)'],
                    'Nilai': [3, salary_pred, 6, 10],
                    'Highlight': ['No', 'Yes', 'No', 'No'],
                })
                fig_bar = px.bar(
                    salary_data, x='Kategori', y='Nilai',
                    color='Highlight',
                    color_discrete_map={'Yes': '#f5c842', 'No': '#2a3050'},
                    text='Nilai',
                )
                fig_bar.update_traces(texttemplate='%{text:.1f} LPA', textposition='outside', textfont_color='#e8e8e8')
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False, height=300,
                    yaxis=dict(gridcolor='#2a2f3d', tickfont={'color': '#7a8099'}, title='LPA', title_font={'color': '#7a8099'}),
                    xaxis=dict(tickfont={'color': '#c8cdd8'}),
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            elif clf_model and not is_placed:
                st.markdown('<div class="info-box"> Estimasi gaji hanya tersedia untuk mahasiswa yang diprediksi <b>Placed</b>. Tingkatkan performa akademik dan skill untuk meningkatkan peluang.</div>', unsafe_allow_html=True)
            elif reg_model is None:
                st.warning("⚠️ Model regresi tidak ditemukan. Jalankan training terlebih dahulu.")

with tab2:
    col_radar, col_stats = st.columns([1, 1])

    with col_radar:
        st.markdown('<div class="section-header">Profil Kemampuan</div>', unsafe_allow_html=True)
        st.plotly_chart(make_radar(input_row), use_container_width=True)

    with col_stats:
        st.markdown('<div class="section-header">Ringkasan Data</div>', unsafe_allow_html=True)

        # Computed scores
        academic_avg = (tenth_pct + twelfth_pct + cgpa * 10) / 3
        total_skill  = coding_skill + comm_skill + apt_skill
        exp_score    = internships * 2 + projects + hackathons

        st.markdown(f'<div class="metric-card"><h4>ACADEMIC AVG</h4><div class="value">{academic_avg:.1f}</div><div class="sub">Rata-rata 10th + 12th + CGPA(norm)</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><h4>TOTAL SKILL SCORE</h4><div class="value">{total_skill}</div><div class="sub">Coding + Comm + Aptitude</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><h4>EXPERIENCE SCORE</h4><div class="value">{exp_score}</div><div class="sub">Internship×2 + Projects + Hackathon</div></div>', unsafe_allow_html=True)
        risk = " Ada Backlog" if backlogs > 0 else " Bersih"
        st.markdown(f'<div class="metric-card"><h4>BACKLOG STATUS</h4><div class="value" style="font-size:1rem">{risk}</div><div class="sub">{backlogs} backlog tercatat</div></div>', unsafe_allow_html=True)

    # Tabel ringkasan input
    st.markdown('<div class="section-header">Semua Input</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        'Fitur': list(input_row.keys()),
        'Nilai': list(input_row.values()),
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown('<div class="section-header">Tentang Sistem Ini</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Tugas 1 – Klasifikasi:</b> Memprediksi apakah seorang mahasiswa akan <b>Placed</b> (mendapat kerja) atau <b>Not Placed</b>. 
    Model terbaik dipilih berdasarkan <b>ROC-AUC</b> tertinggi dari 5 kandidat: Logistic Regression, Random Forest, XGBoost, LightGBM, dan Gradient Boosting.
    </div>
    <div class="info-box">
    <b>Tugas 2 – Regresi:</b> Memprediksi <b>estimasi gaji</b> (dalam LPA) khusus untuk mahasiswa yang diprediksi Placed. 
    Model terbaik dipilih berdasarkan <b>R² tertinggi</b>.
    </div>
    <div class="info-box">
    <b>Feature Engineering:</b> Sistem menghitung fitur turunan secara otomatis: <code>academic_avg</code>, <code>total_skill_score</code>, <code>experience_score</code>, dan <code>high_risk_backlog</code>.
    </div>
    <div class="info-box">
    <b>Model path:</b><br>
    &nbsp;&nbsp;• <code>artifacts/best_clf_model.pkl</code> – model klasifikasi<br>
    &nbsp;&nbsp;• <code>artifacts/best_reg_model.pkl</code> – model regresi
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Cara Penggunaan</div>', unsafe_allow_html=True)
    st.code("""# 1. Latih model terlebih dahulu
python run_experiment.py --task all

# 2. Jalankan aplikasi Streamlit
streamlit run main.py
""", language="bash")

    st.markdown('<div class="section-header">Fitur Input (24 total)</div>', unsafe_allow_html=True)
    feat_df = pd.DataFrame({
        'Tipe': ['Kategorik'] * len(CAT_FEATURES) + ['Numerik'] * 13 + ['Engineered'] * 4,
        'Fitur': CAT_FEATURES + [
            'cgpa', 'backlogs', 'study_hours_per_day', 'attendance_percentage',
            'projects_completed', 'internships_completed', 'coding_skill_rating',
            'communication_skill_rating', 'aptitude_skill_rating',
            'hackathons_participated', 'certifications_count', 'sleep_hours', 'stress_level'
        ] + ['academic_avg', 'total_skill_score', 'experience_score', 'high_risk_backlog'],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
