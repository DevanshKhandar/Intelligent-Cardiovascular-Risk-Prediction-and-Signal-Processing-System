"""
Heart Disease Prediction & Analysis Dashboard
MLA Mini Project — Covers Units 1-5 of the syllabus
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ml_engine import load_heart_data, prepare_data, regression_analysis
from ml_engine import train_classifiers, train_neural_network
from ml_engine import perform_clustering, perform_pca
from views import p1_explorer, p2_regression, p3_classification
from views import p4_neural_net, p5_clustering_pca, p6_predictor, p7_ecg_signal

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Heart Disease ML Dashboard", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS (White Glassmorphism) ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global background — soft gradient */
.stApp {
    background: linear-gradient(135deg, #e0e7ff 0%, #f0f4ff 30%, #fdf2f8 60%, #ede9fe 100%);
    font-family: 'Inter', sans-serif;
}

/* Sidebar — frosted glass */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.55) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.6);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #1e293b !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li { color: #475569; font-size: 13px; }
[data-testid="stSidebar"] .stMarkdown strong { color: #334155; }

/* Main content area glassmorphism blocks */
[data-testid="stVerticalBlock"] > div > div > [data-testid="stVerticalBlock"] {
    background: rgba(255, 255, 255, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.6);
    padding: 8px;
}

/* Metrics — glass cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.8);
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 13px !important; }
[data-testid="stMetricValue"] { color: #1e293b !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: #6366f1 !important; }

/* Headers */
h1, h2, h3, h4 { color: #1e293b !important; }
p, li { color: #475569; }
span { color: #334155; }

/* Expanders — glass */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.7);
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
}

/* Buttons — gradient glass */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none; border-radius: 12px;
    font-weight: 600; font-size: 16px;
    padding: 12px 28px;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
    transition: all 0.3s ease;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(99, 102, 241, 0.45);
}
.stButton > button {
    background: rgba(255, 255, 255, 0.7) !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-radius: 10px;
    color: #334155 !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: rgba(255, 255, 255, 0.9) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
}

/* Radio */
.stRadio > label { color: #1e293b !important; font-weight: 600; }
div[role="radiogroup"] label { color: #475569 !important; }

/* Selectbox / Slider */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label { color: #334155 !important; }

/* Dividers */
hr { border-color: rgba(0, 0, 0, 0.06) !important; }

/* Plotly chart containers */
[data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, 0.35);
    backdrop-filter: blur(8px);
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.5);
    padding: 4px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
}

/* Spinner */
[data-testid="stSpinner"] { color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🫀 HeartML")
    st.markdown("**Heart Disease Prediction & Analysis**")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📊 Data Explorer",
        "📈 Regression (Unit 1)",
        "🎯 Classification (Unit 2)",
        "🧠 Neural Network (Unit 3)",
        "🔬 Clustering & PCA (Unit 4)",
        "⚡ ECG Signal Processing (Unit 5)",
        "🩺 Risk Predictor",
    ], index=0)

    st.markdown("---")
    st.markdown("### 📚 Syllabus Coverage")
    st.markdown("""
    - **Unit 1:** Linear/Polynomial Regression, Regularisation
    - **Unit 2:** KNN, Decision Trees, Logistic Regression, Naïve Bayes
    - **Unit 3:** Neural Network, Backpropagation
    - **Unit 4:** K-Means, GMM, PCA
    - **Unit 5:** SVD Signal Compression
    """)
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#64748b;font-size:11px;'>MLA Mini Project © 2025</p>",
                unsafe_allow_html=True)


# ── Data & Model Loading (cached) ────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_heart_data()

@st.cache_data
def run_all_models(_df):
    df_copy = _df.copy()
    data = prepare_data(df_copy)
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    X_train_s, X_test_s, scaler = data[4], data[5], data[6]

    reg = regression_analysis(df_copy)
    clf = train_classifiers(X_train_s, X_test_s, y_train, y_test)
    nn = train_neural_network(X_train_s, X_test_s, y_train, y_test)
    X_full_scaled = scaler.transform(df_copy.drop('target', axis=1))
    clust = perform_clustering(X_full_scaled, df_copy['target'].values)
    pca = perform_pca(X_full_scaled)

    return reg, clf, nn, clust, pca, y_test, scaler

df = get_data()

with st.spinner("🔄 Training all ML models... Please wait."):
    reg_res, clf_res, nn_res, cluster_res, pca_res, y_test, scaler = run_all_models(df)

# ── Page Routing ──────────────────────────────────────────────────────────────
if "Explorer" in page:
    p1_explorer.render(df)
elif "Regression" in page:
    p2_regression.render(reg_res)
elif "Classification" in page:
    p3_classification.render(clf_res, y_test)
elif "Neural" in page:
    p4_neural_net.render(nn_res)
elif "Clustering" in page:
    p5_clustering_pca.render(cluster_res, pca_res, df)
elif "ECG" in page:
    p7_ecg_signal.render()
elif "Predictor" in page:
    p6_predictor.render(clf_res, nn_res, scaler)
