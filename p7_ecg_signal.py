"""Page 7: ECG Signal Processing (Unit 5)"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import signal
from sklearn.decomposition import TruncatedSVD

def generate_ecg(length=500, noise_level=0.1):
    """Generate a synthetic ECG PQRST complex signal."""
    x = np.linspace(0, 2*np.pi, length)
    
    # Simulate P, Q, R, S, T waves
    p_wave = np.sin(x * 5) * np.exp(-((x - 1.5)**2) / 0.1) * 0.2
    q_wave = -np.sin(x * 10) * np.exp(-((x - 2.8)**2) / 0.05) * 0.3
    r_wave = np.sin(x * 10) * np.exp(-((x - 3.1)**2) / 0.05) * 1.5
    s_wave = -np.sin(x * 10) * np.exp(-((x - 3.4)**2) / 0.05) * 0.4
    t_wave = np.sin(x * 4) * np.exp(-((x - 4.5)**2) / 0.2) * 0.3
    
    # Baseline wander and noise
    baseline = np.sin(x * 0.5) * 0.1
    noise = np.random.normal(0, noise_level, length)
    
    clean_ecg = p_wave + q_wave + r_wave + s_wave + t_wave + baseline
    noisy_ecg = clean_ecg + noise
    
    return x, clean_ecg, noisy_ecg

def render():
    st.markdown("## ⚡ Live ECG Signal Processing")
    st.markdown("**Unit 5:** Signal Compression with SVD (Singular Value Decomposition)")
    st.markdown("Medical devices generate massive amounts of time-series data (like ECGs). SVD is used to compress these signals, removing noise while preserving the critical PQRST cardiac structure.")
    st.markdown("---")
    
    TPL = dict(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
               plot_bgcolor='rgba(0,0,0,0)', font_color='#1e293b')

    # Controls
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 🎛️ Control Panel")
        noise_level = st.slider("Signal Noise Level", 0.0, 0.5, 0.15, 0.05)
        components = st.slider("SVD Components (Compression Level)", 1, 50, 5, 1)
        
        st.info("📉 **How it works:** We stack multiple heartbeats into a matrix. SVD factors this matrix into vectors. By keeping only the top components, we compress the data and eliminate random noise, effectively cleaning the ECG signal.")

    with col2:
        # Generate data: 50 consecutive heartbeats
        n_beats = 50
        length = 200
        ecg_matrix = []
        for _ in range(n_beats):
            _, _, noisy = generate_ecg(length=length, noise_level=noise_level)
            ecg_matrix.append(noisy)
        ecg_matrix = np.array(ecg_matrix) # Shape: (50, 200)
        
        # Apply SVD Compression
        svd = TruncatedSVD(n_components=components, random_state=42)
        compressed_matrix = svd.fit_transform(ecg_matrix) # Compress
        reconstructed_matrix = svd.inverse_transform(compressed_matrix) # Decompress
        
        compression_ratio = (ecg_matrix.size) / (compressed_matrix.size + svd.components_.size)
        
        # Plotting
        st.markdown("### 📈 Signal Compression & Denoising Analysis")
        
        c1, c2 = st.columns(2)
        c1.metric("Original Size (Data Points)", ecg_matrix.size)
        c2.metric("Compression Ratio", f"{compression_ratio:.1f}x", delta="Smaller File Size", delta_color="normal")
        
        # Pick one heartbeat to display
        idx = 0
        fig = go.Figure()
        
        x_axis = np.arange(length)
        fig.add_trace(go.Scatter(x=x_axis, y=ecg_matrix[idx], mode='lines',
                                 name='Raw Noisy ECG', line=dict(color='rgba(239, 68, 68, 0.4)', width=2)))
        
        fig.add_trace(go.Scatter(x=x_axis, y=reconstructed_matrix[idx], mode='lines',
                                 name=f'SVD Reconstructed (k={components})', 
                                 line=dict(color='#6366f1', width=3)))
        
        fig.update_layout(
            title="Real-time ECG SVD Compression",
            xaxis_title="Time (ms)",
            yaxis_title="Amplitude (mV)",
            height=400,
            **TPL
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # Singular Values Plot
    st.markdown("### 📊 Energy Retention (Singular Values)")
    explained_variance = svd.explained_variance_ratio_.sum() * 100
    st.markdown(f"By using **{components} components**, we preserve **{explained_variance:.2f}%** of the signal's energy while discarding high-frequency noise.")
    
    # Run full SVD just for the plot
    svd_full = TruncatedSVD(n_components=min(n_beats, 40), random_state=42)
    svd_full.fit(ecg_matrix)
    
    fig2 = go.Figure(go.Bar(
        x=[f"σ{i+1}" for i in range(len(svd_full.singular_values_))],
        y=svd_full.singular_values_,
        marker_color=['#6366f1' if i < components else '#cbd5e1' for i in range(len(svd_full.singular_values_))]
    ))
    fig2.update_layout(
        title="Singular Values Magnitude (Blue = Kept for Compression)",
        xaxis_title="Singular Value Index",
        yaxis_title="Magnitude",
        height=300,
        **TPL
    )
    st.plotly_chart(fig2, use_container_width=True)
