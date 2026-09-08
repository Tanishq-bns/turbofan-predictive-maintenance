import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroPredict - Turbofan Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom sleek industrial CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .status-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #c3e6cb;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #ffeeba;
    }
    .status-critical {
        background-color: #f8d7da;
        color: #721c24;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Cached Artifacts Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_metadata():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    clf = joblib.load(os.path.join(models_dir, "xgb_classifier.joblib"))
    reg = joblib.load(os.path.join(models_dir, "xgb_regressor.joblib"))
    meta = joblib.load(os.path.join(models_dir, "meta_info.joblib"))
    sample_df = pd.read_csv(os.path.join(models_dir, "sample_engines.csv"))
    return clf, reg, meta, sample_df

clf, reg, meta, sample_df = load_models_and_metadata()
feature_cols = meta['feature_cols']
min_risk = meta['min_risk']
max_risk = meta['max_risk']
risk_threshold = meta['risk_threshold']

# -----------------------------------------------------------------------------
# 3. Core Inference Function
# -----------------------------------------------------------------------------
def predict_health(input_df):
    """
    Takes a dataframe containing feature columns, applies XGBoost models,
    and returns predictions and risk score.
    """
    X = input_df[feature_cols]
    
    # 1. Classification (Degradation Stage: 0-4)
    stage_pred = int(clf.predict(X)[0])
    probabilities = clf.predict_proba(X)[0]
    p_stage_4 = float(probabilities[4])
    
    # 2. Regression (Time to next stage)
    time_to_next = float(reg.predict(X)[0])
    time_to_next = max(0.0, time_to_next)
    
    # 3. Risk Calculation
    raw_risk = p_stage_4 * time_to_next
    normalized_risk = (raw_risk - min_risk) / (max_risk - min_risk + 1e-6)
    normalized_risk = float(np.clip(normalized_risk, 0.0, 1.0))
    
    # 4. Alert Level
    if normalized_risk >= risk_threshold or stage_pred == 4:
        alert_status = "CRITICAL"
    elif normalized_risk >= 0.4 or stage_pred >= 2:
        alert_status = "WARNING"
    else:
        alert_status = "NORMAL"
        
    return {
        'stage': stage_pred,
        'stage_label': meta['stage_labels'].get(stage_pred, f"Stage {stage_pred}"),
        'probabilities': probabilities,
        'p_stage_4': p_stage_4,
        'time_to_next': round(time_to_next, 1),
        'normalized_risk': normalized_risk,
        'alert_status': alert_status
    }

# -----------------------------------------------------------------------------
# 4. Sidebar: Navigation & Controls
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/jet-engine.png", width=70)
st.sidebar.title("AeroPredict AI")
st.sidebar.caption("NASA C-MAPSS Turbofan Health Monitoring")

mode = st.sidebar.radio(
    "Select Operating Mode:",
    ["Fleet Engine Explorer", "Manual Telemetry Simulator", "Model & Sensor Insights"]
)

st.sidebar.divider()

# -----------------------------------------------------------------------------
# MODE 1: Fleet Engine Explorer
# -----------------------------------------------------------------------------
if mode == "Fleet Engine Explorer":
    st.title("✈️ Turbofan Fleet Degradation Explorer")
    st.markdown("""
    Explore historical engine flight lifecycles from the NASA C-MAPSS dataset.
    Select an engine unit and scrub through operational cycles to observe degradation progression in real time.
    """)

    available_units = sorted(sample_df['unit'].unique())
    selected_unit = st.sidebar.selectbox("Select Engine Unit ID:", available_units, index=0)
    
    unit_data = sample_df[sample_df['unit'] == selected_unit].sort_values('time').reset_index(drop=True)
    max_cycles = int(unit_data['time'].max())
    
    selected_cycle = st.sidebar.slider(
        "Scrub Operational Cycle:",
        min_value=1,
        max_value=max_cycles,
        value=min(max_cycles, max(1, int(max_cycles * 0.8))),
        step=1
    )
    
    # Current cycle slice
    cycle_row = unit_data[unit_data['time'] == selected_cycle]
    if cycle_row.empty:
        cycle_row = unit_data.iloc[-1:]

    pred = predict_health(cycle_row)

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Degradation Health Stage", f"Stage {pred['stage']}", delta=meta['stage_short'].get(pred['stage']))
    with col2:
        st.metric("Cycles to Next Stage", f"{pred['time_to_next']} cycles")
    with col3:
        st.metric("Failure Prob P(Stage 4)", f"{pred['p_stage_4'] * 100:.1f} %")
    with col4:
        st.metric("Risk Index", f"{pred['normalized_risk'] * 100:.1f} %")

    # Maintenance Alert Banner
    st.write("")
    if pred['alert_status'] == "CRITICAL":
        st.markdown("""
        <div class="status-critical">
            🚨 <b>CRITICAL MAINTENANCE ALERT TRIGGERED!</b><br>
            Accelerated failure probability detected. Ground engine immediately for hot-section inspection and scheduled blade replacement.
        </div>
        """, unsafe_allow_html=True)
    elif pred['alert_status'] == "WARNING":
        st.markdown("""
        <div class="status-warning">
            ⚠️ <b>MAINTENANCE CAUTION</b><br>
            Moderate degradation detected. Recommend inspection during the next routine A-check turnaround.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-normal">
            ✅ <b>NOMINAL ENGINE HEALTH</b><br>
            Turbofan operating normally within nominal flight envelopes. No immediate maintenance required.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Telemetry Plots
    st.subheader(f"📈 Unit {selected_unit} Telemetry & Health Trajectory")

    # Compute trajectory over all cycles for this unit
    unit_features = unit_data[feature_cols]
    all_stages = clf.predict(unit_features)
    all_times = reg.predict(unit_features)
    all_p4 = clf.predict_proba(unit_features)[:, 4]
    
    raw_risks = all_p4 * np.clip(all_times, 0, None)
    norm_risks = np.clip((raw_risks - min_risk) / (max_risk - min_risk + 1e-6), 0.0, 1.0)
    
    trajectory_df = pd.DataFrame({
        'Cycle': unit_data['time'],
        'Risk Index (%)': norm_risks * 100,
        'Stage': all_stages,
        'HPC Outlet Temp (s3)': unit_data['sensor3'],
        'Fan Speed (s8)': unit_data['sensor8'],
        'HPC Outlet Press (s7)': unit_data['sensor7'],
        'Bypass Ratio (s15)': unit_data['sensor15']
    })

    tab1, tab2, tab3 = st.tabs(["⚠️ Risk Trajectory", "🔥 Temperature & Pressure Sensors", "⚡ Rotational Speed & Bypass"])

    with tab1:
        fig_risk = px.line(
            trajectory_df, 
            x='Cycle', 
            y='Risk Index (%)', 
            title=f"Normalized Risk Index Across Lifespan (Unit {selected_unit})",
            color_discrete_sequence=['#d62728']
        )
        fig_risk.add_hline(y=risk_threshold * 100, line_dash="dash", line_color="orange", annotation_text="Alert Threshold (70%)")
        fig_risk.add_vline(x=selected_cycle, line_dash="dot", line_color="blue", annotation_text=f"Selected Cycle {selected_cycle}")
        fig_risk.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_risk, width='stretch')

    with tab2:
        fig_temp = px.line(
            trajectory_df, 
            x='Cycle', 
            y=['HPC Outlet Temp (s3)', 'HPC Outlet Press (s7)'],
            title=f"Thermodynamic Degradation: HPC Outlet Temp & Pressure (Unit {selected_unit})"
        )
        fig_temp.add_vline(x=selected_cycle, line_dash="dot", line_color="blue")
        fig_temp.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_temp, width='stretch')

    with tab3:
        fig_speed = px.line(
            trajectory_df, 
            x='Cycle', 
            y=['Fan Speed (s8)', 'Bypass Ratio (s15)'],
            title=f"Mechanical Degradation: Fan Speed & Bypass Ratio (Unit {selected_unit})"
        )
        fig_speed.add_vline(x=selected_cycle, line_dash="dot", line_color="blue")
        fig_speed.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_speed, width='stretch')

# -----------------------------------------------------------------------------
# MODE 2: Manual Telemetry Simulator
# -----------------------------------------------------------------------------
elif mode == "Manual Telemetry Simulator":
    st.title("🎛️ Live Sensor Telemetry Simulator")
    st.markdown("Manually inject sensor readings to simulate operating conditions and observe live AI predictions.")

    # Populate default baseline values from median of normal stage
    baseline = sample_df[sample_df['stages'] == 0][feature_cols].median()

    st.subheader("Key Sensor Inputs")
    col_a, col_b, col_c = st.columns(3)

    custom_inputs = {}

    with col_a:
        custom_inputs['operational_set1'] = st.slider("Altitude Setting 1", -0.01, 0.01, float(baseline['operational_set1']), step=0.001)
        custom_inputs['sensor2'] = st.slider("Total Temp at LPC Outlet (T24)", 640.0, 650.0, float(baseline['sensor2']), step=0.1)
        custom_inputs['sensor3'] = st.slider("Total Temp at HPC Outlet (T30)", 1570.0, 1620.0, float(baseline['sensor3']), step=0.5)
        custom_inputs['sensor4'] = st.slider("Total Temp at LPT Outlet (T50)", 1390.0, 1440.0, float(baseline['sensor4']), step=0.5)

    with col_b:
        custom_inputs['operational_set2'] = st.slider("Mach Setting 2", -0.001, 0.001, float(baseline['operational_set2']), step=0.0001)
        custom_inputs['sensor7'] = st.slider("Total Pressure at HPC Outlet (P30)", 548.0, 560.0, float(baseline['sensor7']), step=0.2)
        custom_inputs['sensor8'] = st.slider("Physical Fan Speed (Nf)", 2387.0, 2392.0, float(baseline['sensor8']), step=0.1)
        custom_inputs['sensor9'] = st.slider("Physical Core Speed (Nc)", 9040.0, 9100.0, float(baseline['sensor9']), step=1.0)

    with col_c:
        custom_inputs['operational_set3'] = st.slider("TRA Throttle Setting 3", 90.0, 110.0, float(baseline['operational_set3']), step=1.0)
        custom_inputs['sensor11'] = st.slider("Static Pressure at HPC Outlet (Ps30)", 46.8, 48.5, float(baseline['sensor11']), step=0.05)
        custom_inputs['sensor12'] = st.slider("Ratio of Fuel Flow to Ps30", 518.0, 525.0, float(baseline['sensor12']), step=0.2)
        custom_inputs['sensor15'] = st.slider("Bypass Ratio (BPR)", 8.3, 8.6, float(baseline['sensor15']), step=0.01)

    # Fill remaining unadjusted sensors with baseline values
    full_input_dict = {}
    for col in feature_cols:
        if col in custom_inputs:
            full_input_dict[col] = custom_inputs[col]
        else:
            full_input_dict[col] = float(baseline[col])

    input_df = pd.DataFrame([full_input_dict])
    pred = predict_health(input_df)

    st.divider()
    st.subheader("Simulated Prediction Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Health Stage", f"Stage {pred['stage']}", delta=meta['stage_short'].get(pred['stage']))
    with col2:
        st.metric("Cycles to Next Stage", f"{pred['time_to_next']} cycles")
    with col3:
        st.metric("Failure Prob P(Stage 4)", f"{pred['p_stage_4'] * 100:.1f} %")
    with col4:
        st.metric("Risk Score", f"{pred['normalized_risk'] * 100:.1f} %")

    # Probability Distribution Bar Chart
    prob_df = pd.DataFrame({
        'Stage': [f"Stage {i}: {meta['stage_short'][i]}" for i in range(5)],
        'Probability': pred['probabilities']
    })
    fig_prob = px.bar(
        prob_df, 
        x='Stage', 
        y='Probability', 
        color='Stage', 
        title="Class Probability Distribution",
        range_y=[0, 1]
    )
    st.plotly_chart(fig_prob, width='stretch')

# -----------------------------------------------------------------------------
# MODE 3: Model & Sensor Insights
# -----------------------------------------------------------------------------
elif mode == "Model & Sensor Insights":
    st.title("🧠 Predictive Maintenance Model Architecture")
    st.markdown("""
    ### Pipeline Workflow
    This application utilizes a dual-model architecture:
    1. **XGBoost Classifier**: Multi-class tree booster identifying real-time degradation state (0 to 4) with **98% test accuracy**.
    2. **XGBoost Regressor**: Estimates operational cycles remaining before transitioning into the next degradation stage.
    3. **Composite Risk Engine**: Fuses transition time with the probability of catastrophic failure ($P(\\text{Stage 4})$):
    """)
    st.latex(r"\text{Risk Score} = \frac{P(\text{Stage 4}) \times \hat{T}_{\text{next}} - \text{MinRisk}}{\text{MaxRisk} - \text{MinRisk}}")

    st.subheader("Sensor Dictionary & Descriptions")
    sensor_df = pd.DataFrame(
        list(meta['sensor_descriptions'].items()),
        columns=['Feature Identifier', 'Description / Engineering Measurement']
    )
    st.dataframe(sensor_df, width='stretch', hide_index=True)
