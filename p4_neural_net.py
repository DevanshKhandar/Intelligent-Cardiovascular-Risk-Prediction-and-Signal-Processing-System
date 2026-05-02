"""Page 4: Neural Network — Unit 3"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import plotly.express as px


def render(nn_res):
    st.markdown("## 🧠 Neural Network")
    st.markdown("**Unit 3:** MLP with Backpropagation, Gradient Descent Optimization")
    st.markdown("---")
    TPL = dict(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
               plot_bgcolor='rgba(0,0,0,0)', font_color='#1e293b')

    # Architecture
    st.markdown("### 🏗️ Network Architecture")
    arch = nn_res['architecture']
    arch_str = " → ".join([f"**{n}**" for n in arch])
    st.markdown(f"Input({arch[0]}) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Output(1, Sigmoid)")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{nn_res['accuracy']:.4f}")
    c2.metric("Precision", f"{nn_res['precision']:.4f}")
    c3.metric("Recall", f"{nn_res['recall']:.4f}")
    c4.metric("F1 Score", f"{nn_res['f1']:.4f}")

    c5, c6 = st.columns(2)
    c5.metric("AUC-ROC", f"{nn_res['auc']:.4f}")
    c6.metric("Epochs Trained", nn_res['n_iter'])

    st.markdown("---")

    # Network Visualization
    st.markdown("### 🔗 Network Visualization")
    fig = go.Figure()
    layers = [13, 64, 32, 16, 1]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Hidden 3', 'Output']
    colors = ['#818cf8', '#a78bfa', '#c084fc', '#e879f9', '#f472b6']
    max_nodes_display = [13, 8, 6, 4, 1]
    x_positions = [0, 1, 2, 3, 4]

    for li, (n_display, x_pos) in enumerate(zip(max_nodes_display, x_positions)):
        y_positions = np.linspace(-n_display / 2, n_display / 2, n_display)
        for yi in y_positions:
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[yi], mode='markers+text',
                marker=dict(size=20, color=colors[li], line=dict(width=1, color='white')),
                showlegend=False, hoverinfo='skip'
            ))
        # Connections to next layer
        if li < len(max_nodes_display) - 1:
            next_n = max_nodes_display[li + 1]
            next_y = np.linspace(-next_n / 2, next_n / 2, next_n)
            for y1 in y_positions[::2]:
                for y2 in next_y:
                    fig.add_trace(go.Scatter(
                        x=[x_pos, x_positions[li + 1]], y=[y1, y2], mode='lines',
                        line=dict(color='rgba(99,102,241,0.12)', width=0.5),
                        showlegend=False, hoverinfo='skip'
                    ))
        fig.add_annotation(x=x_pos, y=-max(max_nodes_display) / 2 - 1.5,
                           text=f"{layer_names[li]}<br>({layers[li]})",
                           showarrow=False, font=dict(color='#334155', size=11))
    fig.update_layout(height=450, xaxis=dict(visible=False), yaxis=dict(visible=False), **TPL)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Training Loss Curve (Backpropagation)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📉 Training Loss (Backpropagation)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=nn_res['loss_curve'], mode='lines',
                                 line=dict(color='#f97316', width=2), name='Training Loss'))
        fig.update_layout(xaxis_title='Epoch', yaxis_title='Loss', **TPL)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if nn_res.get('val_scores'):
            st.markdown("### 📈 Validation Accuracy")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=nn_res['val_scores'], mode='lines',
                                     line=dict(color='#4ade80', width=2), name='Val Accuracy'))
            fig.update_layout(xaxis_title='Epoch', yaxis_title='Accuracy', **TPL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("### 📈 ROC Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nn_res['fpr'], y=nn_res['tpr'], mode='lines',
                                     line=dict(color='#4ade80', width=2),
                                     name=f"AUC={nn_res['auc']:.3f}"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     line=dict(color='gray', dash='dash')))
            fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', **TPL)
            st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    st.markdown("### 🔢 Confusion Matrix")
    cm = nn_res['cm']
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Purples',
                    x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'])
    fig.update_layout(height=350, **TPL)
    st.plotly_chart(fig, use_container_width=True)

    # ── CRAZY FEATURE: 3D Loss Landscape ──
    st.markdown("---")
    st.markdown("### 🌌 3D Neural Network Loss Landscape")
    st.markdown("Visualizing the non-convex cost function space $J(W_1, W_2)$ that Gradient Descent navigates during backpropagation.")
    
    # Simulate a complex non-convex loss landscape (Ackley-like or Bowl with ripples)
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 + Y**2) - np.cos(3*np.pi*X) - np.cos(3*np.pi*Y) + 2 # Rippled bowl
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Plasma', opacity=0.9)])
    
    # Add a gradient descent path (simulated)
    path_x = [-2.5, -2.0, -1.2, -0.5, 0.0, 0.1]
    path_y = [2.5,  1.5,  0.8,  0.2, 0.1, 0.0]
    path_z = [(px**2 + py**2) - np.cos(3*np.pi*px) - np.cos(3*np.pi*py) + 2 + 1 for px, py in zip(path_x, path_y)]
    
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines+markers',
        marker=dict(size=6, color='#22c55e', symbol='diamond'),
        line=dict(color='#22c55e', width=4),
        name='Gradient Descent Path'
    ))
    
    fig.update_layout(
        title='Optimization Surface & Gradient Descent Trajectory',
        scene=dict(
            xaxis_title='Weight 1',
            yaxis_title='Weight 2',
            zaxis_title='Loss J(W)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600,
        **TPL
    )
    st.plotly_chart(fig, use_container_width=True)
