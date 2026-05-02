"""Page 6: Interactive Patient Risk Predictor"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from ml_engine import FEATURE_INFO


def render(clf_results, nn_res, scaler):
    st.markdown("## 🩺 Patient Risk Predictor")
    st.markdown("Enter patient details to get heart disease risk predictions from all trained models.")
    st.markdown("---")
    TPL = dict(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
               plot_bgcolor='rgba(0,0,0,0)', font_color='#1e293b')

    # Input form
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 20, 90, 55)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                          format_func=lambda x: ['Typical Angina', 'Atypical', 'Non-anginal', 'Asymptomatic'][x])
        trestbps = st.slider("Resting Blood Pressure", 90, 200, 130)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 250)
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Rest ECG", [0, 1, 2],
                               format_func=lambda x: ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'][x])
        thalach = st.slider("Max Heart Rate", 70, 210, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.slider("ST Depression", 0.0, 7.0, 1.0, 0.1)
    with col3:
        slope = st.selectbox("ST Slope", [0, 1, 2],
                             format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
        ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                            format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown'][x])

    # Predict button
    if st.button("🔮 Predict Risk", use_container_width=True, type="primary"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_data)

        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        models_to_predict = {k: v for k, v in clf_results.items() if not k.startswith('_')}
        predictions = {}
        probabilities = {}

        for name, res in models_to_predict.items():
            model = res['model']
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, 'predict_proba') else pred
            predictions[name] = pred
            probabilities[name] = prob

        # Neural Network
        nn_model = nn_res['model']
        nn_pred = nn_model.predict(input_scaled)[0]
        nn_prob = nn_model.predict_proba(input_scaled)[0][1]
        predictions['Neural Network'] = nn_pred
        probabilities['Neural Network'] = nn_prob

        # Display results
        cols = st.columns(len(predictions))
        for i, (name, pred) in enumerate(predictions.items()):
            prob = probabilities[name]
            with cols[i]:
                emoji = "🔴" if pred == 1 else "🟢"
                status = "At Risk" if pred == 1 else "Healthy"
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.6); backdrop-filter:blur(10px);
                     border-radius:14px; padding:16px; text-align:center;
                     border: 1px solid rgba(255,255,255,0.8); box-shadow: 0 4px 16px rgba(0,0,0,0.04);">
                    <div style="font-size:28px;">{emoji}</div>
                    <div style="font-size:12px; color:#64748b; margin-top:4px;">{name}</div>
                    <div style="font-size:18px; font-weight:bold; color:{'#dc2626' if pred else '#16a34a'};">
                        {status}</div>
                    <div style="font-size:13px; color:#475569;">Confidence: {prob:.1%}</div>
                </div>""", unsafe_allow_html=True)

        # Gauge chart for average risk
        avg_prob = np.mean(list(probabilities.values()))
        st.markdown("---")
        st.markdown("### 🎯 Overall Risk Assessment")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_prob * 100,
            title={'text': "Heart Disease Risk Score", 'font': {'color': '#1e293b', 'size': 18}},
            number={'suffix': '%', 'font': {'color': '#1e293b', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#475569'},
                'bar': {'color': '#6366f1'},
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(74, 222, 128, 0.25)'},
                    {'range': [30, 60], 'color': 'rgba(251, 191, 36, 0.25)'},
                    {'range': [60, 100], 'color': 'rgba(248, 113, 113, 0.25)'},
                ],
                'threshold': {
                    'line': {'color': '#6366f1', 'width': 3},
                    'thickness': 0.8, 'value': avg_prob * 100
                }
            }
        ))
        fig.update_layout(height=320, **TPL)
        st.plotly_chart(fig, use_container_width=True)

        # Verdict
        if avg_prob < 0.3:
            st.success("✅ **Low Risk** — The patient appears to be at low risk for heart disease.")
        elif avg_prob < 0.6:
            st.warning("⚠️ **Moderate Risk** — Further medical evaluation is recommended.")
        else:
            st.error("🚨 **High Risk** — The patient shows high risk indicators. Immediate medical consultation advised.")

        # ── CRAZY FEATURE: What-If Optimizer ──
        st.markdown("---")
        st.markdown("### 🪄 AI Health Optimizer (What-If Analysis)")
        st.markdown("Simulate how improving specific health metrics would reduce this patient's risk.")
        
        scenarios = {'Current Baseline': input_data[0].copy()}
        
        # Scenario 1: Blood Pressure
        if trestbps > 120:
            s1 = input_data[0].copy()
            s1[3] = 120 # index 3 is trestbps
            scenarios['Lowered Blood Pressure (120)'] = s1
            
        # Scenario 2: Cholesterol
        if chol > 200:
            s2 = input_data[0].copy()
            s2[4] = 200 # index 4 is chol
            scenarios['Lowered Cholesterol (200)'] = s2
            
        # Scenario 3: Exercise
        if thalach < 160:
            s3 = input_data[0].copy()
            s3[7] = 160 # index 7 is thalach
            scenarios['Improved Fitness (Max HR 160)'] = s3
            
        # Scenario 4: All combined
        if len(scenarios) > 1:
            s_all = input_data[0].copy()
            s_all[3] = min(s_all[3], 120)
            s_all[4] = min(s_all[4], 200)
            s_all[7] = max(s_all[7], 160)
            scenarios['Optimal Lifestyle Changes'] = s_all
            
        scenario_names = list(scenarios.keys())
        scenario_risks = []
        
        for name, data_arr in scenarios.items():
            s_scaled = scaler.transform([data_arr])
            s_prob = nn_model.predict_proba(s_scaled)[0][1] * 100
            scenario_risks.append(s_prob)
            
        # Plot Waterfall / Bar
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=scenario_names, 
            y=scenario_risks,
            text=[f"{r:.1f}%" for r in scenario_risks],
            textposition='auto',
            marker_color=['#ef4444' if i==0 else '#3b82f6' if i<len(scenario_names)-1 else '#22c55e' for i in range(len(scenario_names))],
            opacity=0.85
        ))
        fig.update_layout(
            title='Simulated Risk Reduction Strategy',
            yaxis_title='Predicted Risk (%)',
            yaxis_range=[0, max(100, max(scenario_risks)+10)],
            **TPL
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **XAI Note:** This counterfactual simulation uses the Neural Network model to recalculate risk under optimized hypothetical conditions, providing actionable health intelligence.")

