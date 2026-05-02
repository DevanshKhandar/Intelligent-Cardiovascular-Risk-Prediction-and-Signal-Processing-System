"""Page 5: Clustering & PCA — Units 4 & 5"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def render(cluster_res, pca_res, df):
    st.markdown("## 🔬 Clustering & PCA")
    st.markdown("**Unit 4:** K-Means, GMM  |  **Unit 5:** PCA Dimensionality Reduction")
    st.markdown("---")
    TPL = dict(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
               plot_bgcolor='rgba(0,0,0,0)', font_color='#1e293b')

    # ── K-Means ──
    st.markdown("### 📍 K-Means Clustering")

    # Elbow Method
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cluster_res['K_range'], y=cluster_res['inertias'],
                                 mode='lines+markers', line=dict(color='#818cf8', width=2),
                                 marker=dict(size=8)))
        fig.add_vline(x=3, line_dash='dash', line_color='#f97316',
                      annotation_text='K=3 (chosen)')
        fig.update_layout(title='Elbow Method', xaxis_title='Number of Clusters (K)',
                          yaxis_title='Inertia (WCSS)', **TPL)
        st.plotly_chart(fig, use_container_width=True)

    # Cluster Visualization (2D PCA)
    with col2:
        plot_df = pd.DataFrame({'PC1': pca_res['X_2d'][:, 0], 'PC2': pca_res['X_2d'][:, 1],
                                'Cluster': cluster_res['km_labels'].astype(str)})
        fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                         color_discrete_sequence=['#818cf8', '#f472b6', '#4ade80'],
                         title='K-Means Clusters (PCA 2D)')
        fig.update_layout(**TPL)
        st.plotly_chart(fig, use_container_width=True)

    # Cluster Profiles
    st.markdown("#### 📋 Cluster Profiles (Mean Feature Values)")
    profile_df = df.drop('target', axis=1).copy()
    profile_df['Cluster'] = cluster_res['km_labels']
    profiles = profile_df.groupby('Cluster').mean().round(2)
    # Add disease rate
    profile_df['target'] = df['target'].values
    profiles['Disease Rate %'] = (profile_df.groupby('Cluster')['target'].mean() * 100).round(1)
    profiles['Count'] = profile_df.groupby('Cluster').size()
    st.dataframe(profiles.round(2), use_container_width=True)

    st.markdown("---")

    # ── GMM ──
    st.markdown("### 🌀 Gaussian Mixture Model")
    col1, col2 = st.columns(2)
    with col1:
        gmm_df = pd.DataFrame({'PC1': pca_res['X_2d'][:, 0], 'PC2': pca_res['X_2d'][:, 1],
                                'Component': cluster_res['gmm_labels'].astype(str)})
        fig = px.scatter(gmm_df, x='PC1', y='PC2', color='Component',
                         color_discrete_sequence=['#fbbf24', '#22d3ee', '#e879f9'],
                         title='GMM Components (PCA 2D)')
        fig.update_layout(**TPL)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Compare K-Means vs GMM vs True labels
        true_df = pd.DataFrame({'PC1': pca_res['X_2d'][:, 0], 'PC2': pca_res['X_2d'][:, 1],
                                'Label': cluster_res['y_true'].astype(str)})
        fig = px.scatter(true_df, x='PC1', y='PC2', color='Label',
                         color_discrete_map={'0': '#4ade80', '1': '#f87171'},
                         title='True Labels (PCA 2D)')
        fig.update_layout(**TPL)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── PCA ──
    st.markdown("### 📐 PCA Dimensionality Reduction")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Components for 95% Variance", pca_res['n_95'])
    with c2:
        st.metric("Total Features", len(pca_res['explained']))

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f'PC{i+1}' for i in range(len(pca_res['explained']))],
                             y=pca_res['explained'], name='Individual',
                             marker_color='#818cf8', opacity=0.7))
        fig.add_trace(go.Scatter(x=[f'PC{i+1}' for i in range(len(pca_res['cumulative']))],
                                 y=pca_res['cumulative'], mode='lines+markers',
                                 name='Cumulative', line=dict(color='#f97316', width=2)))
        fig.add_hline(y=0.95, line_dash='dash', line_color='#4ade80',
                      annotation_text='95% threshold')
        fig.update_layout(title='Explained Variance Ratio', yaxis_title='Variance Ratio', **TPL)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 3D PCA scatter
        pca3d_df = pd.DataFrame({
            'PC1': pca_res['X_3d'][:, 0], 'PC2': pca_res['X_3d'][:, 1],
            'PC3': pca_res['X_3d'][:, 2], 'Target': df['target'].astype(str)
        })
        fig = px.scatter_3d(pca3d_df, x='PC1', y='PC2', z='PC3', color='Target',
                            color_discrete_map={'0': '#4ade80', '1': '#f87171'},
                            title='3D PCA Projection', opacity=0.7)
        fig.update_layout(height=450, **TPL)
        st.plotly_chart(fig, use_container_width=True)
