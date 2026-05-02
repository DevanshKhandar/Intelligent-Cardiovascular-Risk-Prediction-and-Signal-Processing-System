"""Page 3: Classification Models — Unit 2"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def render(clf_results, y_test):
    st.markdown("## 🎯 Classification Models")
    st.markdown("**Unit 2:** KNN, Decision Trees, Logistic Regression, Naïve Bayes")
    st.markdown("---")
    TPL = dict(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
               plot_bgcolor='rgba(0,0,0,0)', font_color='#1e293b')

    models = {k: v for k, v in clf_results.items() if not k.startswith('_')}

    # --- Metrics comparison ---
    st.markdown("### 📊 Model Performance Comparison")
    metrics_df = pd.DataFrame({
        name: {'Accuracy': r['accuracy'], 'Precision': r['precision'],
               'Recall': r['recall'], 'F1 Score': r['f1']}
        for name, r in models.items()
    }).T
    st.dataframe(metrics_df.round(4), use_container_width=True)

    # Bar chart
    fig = go.Figure()
    colors = ['#818cf8', '#f472b6', '#34d399', '#fbbf24']
    for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
        fig.add_trace(go.Bar(
            name=metric.capitalize(), x=list(models.keys()),
            y=[m[metric] for m in models.values()],
            marker_color=colors[i], opacity=0.85
        ))
    fig.update_layout(barmode='group', title='Model Metrics Comparison',
                      yaxis_title='Score', yaxis_range=[0, 1], **TPL)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Confusion Matrices ---
    st.markdown("### 🔢 Confusion Matrices")
    cols = st.columns(2)
    for idx, (name, res) in enumerate(models.items()):
        with cols[idx % 2]:
            cm = res['cm']
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                            x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'],
                            title=name, aspect='equal')
            fig.update_layout(height=320, **TPL)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- ROC Curves ---
    st.markdown("### 📉 ROC Curves")
    fig = go.Figure()
    roc_colors = ['#818cf8', '#f472b6', '#34d399', '#fbbf24']
    for i, (name, res) in enumerate(models.items()):
        if 'fpr' in res:
            fig.add_trace(go.Scatter(
                x=res['fpr'], y=res['tpr'], mode='lines',
                name=f"{name} (AUC={res['auc']:.3f})",
                line=dict(color=roc_colors[i], width=2)
            ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                             line=dict(color='gray', dash='dash', width=1)))
    fig.update_layout(title='ROC Curves', xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate', **TPL)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- KNN K Analysis ---
    st.markdown("### 🔍 KNN: Optimal K Selection")
    ka = clf_results['_knn_k_analysis']
    best_k = ka['k_range'][np.argmax(ka['scores'])]
    st.metric("Best K", best_k, f"Accuracy: {max(ka['scores']):.4f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ka['k_range'], y=ka['scores'], mode='lines+markers',
                             line=dict(color='#818cf8', width=2),
                             marker=dict(size=8, color='#818cf8')))
    fig.add_vline(x=best_k, line_dash='dash', line_color='#f97316',
                  annotation_text=f'Best K={best_k}')
    fig.update_layout(title='Accuracy vs K', xaxis_title='K', yaxis_title='Accuracy', **TPL)
    st.plotly_chart(fig, use_container_width=True)

    # ── CRAZY FEATURE: Explainable AI (XAI) ──
    st.markdown("---")
    st.markdown("### 🕵️ Explainable AI (XAI): Feature Importance")
    st.markdown("Understanding *why* the models make their predictions is critical in healthcare. Below is the decision breakdown for the Tree and Linear models.")
    
    col1, col2 = st.columns(2)
    features = ['Age', 'Sex', 'Chest Pain', 'BP', 'Cholesterol', 'Fasting BS', 'ECG', 'Max HR', 'Exercise Angina', 'ST Depress', 'Slope', 'Vessels', 'Thal']
    
    with col1:
        dt = models.get('Decision Tree', {}).get('model')
        if dt and hasattr(dt, 'feature_importances_'):
            fi = dt.feature_importances_
            idx = np.argsort(fi)[-8:] # Top 8 features
            fig = px.bar(x=fi[idx], y=np.array(features)[idx], orientation='h', 
                         title='Decision Tree: Top Decision Nodes',
                         color=fi[idx], color_continuous_scale='Viridis')
            fig.update_layout(xaxis_title='Gini Importance', yaxis_title='', **TPL)
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        lr = models.get('Logistic Regression', {}).get('model')
        if lr and hasattr(lr, 'coef_'):
            coef = lr.coef_[0]
            idx = np.argsort(np.abs(coef))[-8:]
            colors = ['#ef4444' if c < 0 else '#22c55e' for c in coef[idx]]
            fig = go.Figure(go.Bar(
                x=coef[idx], y=np.array(features)[idx], orientation='h',
                marker_color=colors
            ))
            fig.update_layout(title='Logistic Regression: Top Feature Weights',
                              xaxis_title='Coefficient Weight (Green=Increases Risk, Red=Decreases)', 
                              yaxis_title='', **TPL)
            st.plotly_chart(fig, use_container_width=True)
