"""Page 1: Data Explorer & EDA"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from ml_engine import FEATURE_INFO

TPL = dict(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
           plot_bgcolor='rgba(0,0,0,0)', font_color='#1e293b')


def render(df):
    st.markdown("## 📊 Data Explorer")
    st.markdown("Explore the UCI Heart Disease dataset — 303 patients, 14 clinical features.")

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(df))
    c2.metric("Features", df.shape[1] - 1)
    c3.metric("Heart Disease", int(df['target'].sum()))
    c4.metric("Healthy", int((df['target'] == 0).sum()))

    # Dataset preview
    with st.expander("📋 Dataset Preview", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # Statistical summary
    with st.expander("📈 Statistical Summary", expanded=False):
        st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("---")

    # Target Distribution
    col1, col2 = st.columns(2)
    with col1:
        counts = df['target'].value_counts()
        fig = px.pie(values=counts.values, names=['Healthy', 'Heart Disease'],
                     color_discrete_sequence=['#4ade80', '#f87171'],
                     title="Target Distribution", hole=0.45)
        fig.update_layout(**TPL)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x='age', color='target', barmode='overlay',
                           color_discrete_map={0: '#4ade80', 1: '#f87171'},
                           labels={'target': 'Heart Disease'}, title="Age Distribution by Target")
        fig.update_layout(**TPL)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.markdown("### 🔥 Feature Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                    aspect='auto', zmin=-1, zmax=1)
    fig.update_layout(height=550, **TPL)
    st.plotly_chart(fig, use_container_width=True)

    # Feature distributions
    st.markdown("### 📊 Feature Distributions")
    feat = st.selectbox("Select Feature", [c for c in df.columns if c != 'target'],
                        format_func=lambda x: FEATURE_INFO.get(x, {}).get('label', x))
    fig = px.histogram(df, x=feat, color='target', barmode='overlay', marginal='box',
                       color_discrete_map={0: '#4ade80', 1: '#f87171'},
                       labels={'target': 'Heart Disease'},
                       title=f"Distribution of {FEATURE_INFO.get(feat, {}).get('label', feat)}")
    fig.update_layout(**TPL)
    st.plotly_chart(fig, use_container_width=True)
