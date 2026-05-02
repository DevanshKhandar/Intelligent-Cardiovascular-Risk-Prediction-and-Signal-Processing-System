"""Page 2: Regression Analysis — Unit 1"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def render(reg):
    st.markdown("## 📈 Regression Analysis")
    st.markdown("**Unit 1:** Linear Regression, Polynomial Curve Fitting, Gradient Descent, Regularisation")
    st.markdown("---")
    TPL = dict(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
               plot_bgcolor='rgba(0,0,0,0)', font_color='#1e293b')

    # --- Linear Regression ---
    st.markdown("### 1. Linear Regression: Age → Max Heart Rate")
    lr = reg['linear']
    col1, col2, col3 = st.columns(3)
    col1.metric("Slope (β₁)", f"{lr['coef']:.3f}")
    col2.metric("Intercept (β₀)", f"{lr['intercept']:.2f}")
    col3.metric("R² Score", f"{lr['r2']:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lr['X'], y=lr['y'], mode='markers', name='Data',
                             marker=dict(color='#818cf8', size=5, opacity=0.6)))
    x_line = np.linspace(lr['X'].min(), lr['X'].max(), 100)
    fig.add_trace(go.Scatter(x=x_line, y=lr['coef'] * x_line + lr['intercept'],
                             mode='lines', name='Best Fit', line=dict(color='#f97316', width=3)))
    fig.update_layout(xaxis_title='Age', yaxis_title='Max Heart Rate',
                      title='Linear Regression Fit', **TPL)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Polynomial Curve Fitting ---
    st.markdown("### 2. Polynomial Curve Fitting")
    poly = reg['polynomial']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lr['X'], y=lr['y'], mode='markers', name='Data',
                             marker=dict(color='#818cf8', size=4, opacity=0.4)))
    colors = ['#f97316', '#22d3ee', '#a78bfa', '#f472b6', '#34d399']
    for i, deg in enumerate(range(1, 6)):
        p = poly[deg]
        fig.add_trace(go.Scatter(x=p['X_sorted'], y=p['y_pred_sorted'], mode='lines',
                                 name=f'Degree {deg} (R²={p["r2"]:.4f})',
                                 line=dict(color=colors[i], width=2)))
    fig.update_layout(title='Polynomial Curve Fitting Comparison',
                      xaxis_title='Age', yaxis_title='Max Heart Rate', **TPL)
    st.plotly_chart(fig, use_container_width=True)

    # R² comparison
    r2_vals = [poly[d]['r2'] for d in range(1, 6)]
    fig2 = px.bar(x=[f'Degree {d}' for d in range(1, 6)], y=r2_vals,
                  color=r2_vals, color_continuous_scale='Viridis', title='R² Score vs Polynomial Degree')
    fig2.update_layout(xaxis_title='Degree', yaxis_title='R² Score', **TPL)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # --- Gradient Descent ---
    st.markdown("### 3. Gradient Descent Convergence")
    st.markdown("Cost function J(w,b) = ½·mean((ŷ - y)²) minimized iteratively.")
    gd = reg['gradient_descent']
    fig = go.Figure()
    gd_colors = {'0.001': '#f87171', '0.01': '#fbbf24', '0.1': '#4ade80'}
    for lr_rate, costs in gd.items():
        fig.add_trace(go.Scatter(y=costs, mode='lines', name=f'α = {lr_rate}',
                                 line=dict(color=gd_colors.get(str(lr_rate), '#818cf8'), width=2)))
    fig.update_layout(title='Gradient Descent: Cost vs Iterations',
                      xaxis_title='Iteration', yaxis_title='Cost J(w,b)', **TPL)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Regularisation ---
    st.markdown("### 4. Ridge vs Lasso Regularisation")
    rv = reg['regularization']
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for i, alpha in enumerate(rv['alphas']):
            fig.add_trace(go.Bar(x=rv['features'], y=rv['ridge'][i]['coefs'],
                                 name=f'α={alpha}', opacity=0.7))
        fig.update_layout(title='Ridge Coefficients', barmode='group',
                          xaxis_title='Feature', yaxis_title='Coefficient', **TPL, height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        for i, alpha in enumerate(rv['alphas']):
            fig.add_trace(go.Bar(x=rv['features'], y=rv['lasso'][i]['coefs'],
                                 name=f'α={alpha}', opacity=0.7))
        fig.update_layout(title='Lasso Coefficients', barmode='group',
                          xaxis_title='Feature', yaxis_title='Coefficient', **TPL, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # R² vs Alpha
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rv['alphas'], y=[r['r2'] for r in rv['ridge']],
                             mode='lines+markers', name='Ridge', line=dict(color='#22d3ee', width=2)))
    fig.add_trace(go.Scatter(x=rv['alphas'], y=[r['r2'] for r in rv['lasso']],
                             mode='lines+markers', name='Lasso', line=dict(color='#f472b6', width=2)))
    fig.update_layout(title='R² Score vs Regularisation Strength (α)',
                      xaxis_title='α (log scale)', yaxis_title='R²', xaxis_type='log', **TPL)
    st.plotly_chart(fig, use_container_width=True)
