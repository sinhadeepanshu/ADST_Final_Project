import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ==============================================================================
# 1. PAGE CONFIGURATION & LAYOUT SETUP
# ==============================================================================
st.set_page_config(
    page_title="AI-Based Predictive Maintenance and Machine Failure Prediction Using Sensor Data",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CUSTOM ENTERPRISE CSS STYLING & ADJUSTED BACKGROUND IMAGE VISIBILITY
# ==============================================================================
st.markdown("""
<style>
    /* Background image with lighter gradient overlay so the factory image is clearly visible */
    .stApp {
        background: linear-gradient(rgba(3, 7, 18, 0.75), rgba(3, 7, 18, 0.82)), 
                    url('https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?q=80&w=1920&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f3f4f6;
        font-family: 'Inter', sans-serif;
    }
    .platform-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 27, 75, 0.85) 50%, rgba(2, 132, 199, 0.85) 100%);
        backdrop-filter: blur(10px);
        padding: 28px;
        border-radius: 16px;
        border: 1px solid rgba(49, 46, 129, 0.6);
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.3);
    }
    .platform-header h1 {
        color: #38bdf8 !important;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        line-height: 1.3;
    }
    .platform-header p {
        color: #cbd5e1 !important;
        font-size: 14px;
        margin-top: 6px;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(11, 15, 25, 0.88);
        backdrop-filter: blur(10px);
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. SECURE AUTHENTICATION SYSTEM
# ==============================================================================
def hash_pass(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_pass(password, hashed):
    return hash_pass(password) == hashed

USERS_DB = {
    "admin": {
        "pass": hash_pass("admin123"),
        "role": "Super Administrator",
        "dept": "Operations & IT"
    },
    "engineer": {
        "pass": hash_pass("pass123"),
        "role": "Reliability Engineer",
        "dept": "Plant Maintenance"
    }
}

# ==============================================================================
# 4. SESSION STATE MANAGEMENT
# ==============================================================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "role" not in st.session_state:
    st.session_state["role"] = ""
if "dept" not in st.session_state:
    st.session_state["dept"] = ""
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = pd.DataFrame(columns=[
        "Timestamp", "User", "Machine Type", "Air Temp (K)", "Process Temp (K)", 
        "RPM", "Torque (Nm)", "Tool Wear (min)", "Failure Risk (%)", "Health Score (%)", "Status"
    ])
if "copilot_messages" not in st.session_state:
    st.session_state["copilot_messages"] = [
        {"role": "assistant", "content": "Hello! I am your AI Maintenance Assistant. Ask me anything regarding machine telemetry or troubleshooting."}
    ]

# ==============================================================================
# 5. LOGIN SCREEN
# ==============================================================================
def render_login():
    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #38bdf8;'>🔐 Enterprise Portal Login</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8;'>AI-Based Predictive Maintenance and Machine Failure Prediction Using Sensor Data</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            uid = st.text_input("Username", placeholder="e.g. admin or engineer")
            pwd = st.text_input("Password", type="password", placeholder="••••••••")
            submit = st.form_submit_button("Sign In Securely", use_container_width=True, type="primary")
            
            if submit:
                clean_uid = uid.strip().lower()
                if clean_uid in USERS_DB and verify_pass(pwd, USERS_DB[clean_uid]["pass"]):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = clean_uid
                    st.session_state["role"] = USERS_DB[clean_uid]["role"]
                    st.session_state["dept"] = USERS_DB[clean_uid]["dept"]
                    st.success("Login Successful! Loading platform...")
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password. Please try again.")
                    
        with st.expander("📌 Demo Credentials"):
            st.markdown("""
            * **Admin:** `admin` / `admin123`
            * **Engineer:** `engineer` / `pass123`
            """)

# ==============================================================================
# 6. MAIN APP DASHBOARD & TOP NAVIGATION BAR
# ==============================================================================
def render_dashboard():
    with st.sidebar:
        st.markdown(f"### 👤 Profile: **{st.session_state['username'].upper()}**")
        st.info(f"**Role:** `{st.session_state['role']}`\n\n**Dept:** `{st.session_state['dept']}`")
        st.markdown("---")
        st.markdown("### 🔌 Session Control")
        if st.button("🚪 Logout Securely", use_container_width=True, type="secondary"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.success("Logged out safely.")
            st.rerun()

    # PROMINENT TOP NAVIGATION BAR (TABS)
    nav_tab1, nav_tab2, nav_tab3, nav_tab4, nav_tab5, nav_tab6, nav_tab7 = st.tabs([
        "🚀 AI Prediction", 
        "📡 Live IoT Stream",
        "📊 Fleet Overview", 
        "📈 Trend Analytics", 
        "🤖 AI Copilot", 
        "📋 Audit Logs", 
        "🛡️ Settings"
    ])

    # ================= TAB 1: AI PREDICTION & DIAGNOSTICS =================
    with nav_tab1:
        st.markdown("""
        <div class="platform-header">
            <h1>AI-Based Predictive Maintenance and Machine Failure Prediction Using Sensor Data</h1>
            <p>Configure machine sensor parameters below and click the prediction button to analyze failure risk instantly.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("prediction_form"):
            st.markdown("### 🎛️ Configure Machine Telemetry Parameters")
            
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                machine_type = st.selectbox("Machine Grade Type", ["Low (L)", "Medium (M)", "High (H)"])
                air_temp = st.slider("Air Temperature [K]", 250.0, 350.0, 298.0, 0.5)
            with f_col2:
                process_temp = st.slider("Process Temperature [K]", 250.0, 400.0, 310.0, 0.5)
                rpm = st.slider("Rotational Speed [RPM]", 1000, 3000, 1550, 10)
            with f_col3:
                torque = st.slider("Operational Torque [Nm]", 0.0, 80.0, 42.0, 0.5)
                tool_wear = st.slider("Cumulative Tool Wear [min]", 0, 300, 110, 1)
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_action = st.form_submit_button("🚀 RUN PREDICTIVE AI MODEL", use_container_width=True, type="primary")

        if predict_action:
            temp_diff = process_temp - air_temp
            base_risk = (tool_wear / 300.0) * 35.0 + (torque / 80.0) * 30.0 + (abs(temp_diff - 10.0) / 20.0) * 20.0 + (abs(rpm - 1500) / 1500.0) * 15.0
            
            if machine_type == "High (H)":
                base_risk += 12.0
            elif machine_type == "Medium (M)":
                base_risk += 5.0

            failure_prob = float(np.clip(base_risk + np.random.uniform(-1.0, 1.0), 1.0, 99.0))
            health_score = float(max(1.0, 100.0 - failure_prob))

            if failure_prob < 30.0:
                status = "OPTIMAL (Normal Operations)"
            elif failure_prob < 70.0:
                status = "WARNING (Elevated Thermal/Mechanical Stress)"
            else:
                status = "CRITICAL (High Failure Risk - Immediate Action)"

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 📊 Prediction & Diagnostic Results")

            res1, res2, res3 = st.columns(3)
            res1.metric("🔴 Failure Probability", f"{failure_prob:.2f}%")
            res2.metric("🟢 Asset Health Score", f"{health_score:.2f}%")
            res3.metric("⚡ Diagnostic Status", status.split(" ")[0])

            st.markdown("<br>", unsafe_allow_html=True)

            diag_col1, diag_col2 = st.columns(2)

            with diag_col1:
                st.markdown("#### 🛡️ Health vs Failure Risk Ratio")
                fig1, ax1 = plt.subplots(figsize=(5, 3))
                fig1.patch.set_facecolor('#0b0f19')
                ax1.set_facecolor('#0b0f19')
                
                cats = ['Health Score', 'Failure Risk']
                vals = [health_score, failure_prob]
                colors = ['#10b981' if health_score >= 50 else '#f59e0b', '#ef4444']
                
                bars = ax1.barh(cats, vals, color=colors, height=0.4)
                ax1.set_xlim(0, 100)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.spines['left'].set_color('#374151')
                ax1.spines['bottom'].set_color('#374151')
                ax1.tick_params(colors='#9ca3af', labelsize=10)
                ax1.set_xlabel('Percentage (%)', color='#9ca3af', fontsize=10)
                
                for bar in bars:
                    w = bar.get_width()
                    ax1.text(w + 2, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', ha='left', color='#ffffff', fontweight='bold', fontsize=10)
                st.pyplot(fig1)

            with diag_col2:
                st.markdown("#### ⚙️ Feature Stress Contribution Weights")
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                fig2.patch.set_facecolor('#0b0f19')
                ax2.set_facecolor('#0b0f19')
                
                features = ['Tool Wear', 'Torque', 'Temp Delta', 'RPM Deviation']
                weights = [
                    (tool_wear / 300.0) * 100,
                    (torque / 80.0) * 100,
                    (abs(temp_diff) / 40.0) * 100,
                    (abs(rpm - 1500) / 1500.0) * 100
                ]
                
                sns.barplot(x=weights, y=features, ax=ax2, palette='Blues_r', orient='h')
                ax2.set_xlim(0, 100)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_color('#374151')
                ax2.spines['bottom'].set_color('#374151')
                ax2.tick_params(colors='#9ca3af', labelsize=10)
                ax2.set_xlabel('Relative Impact (%)', color='#9ca3af', fontsize=10)
                ax2.set_ylabel('', color='#9ca3af')
                st.pyplot(fig2)

            st.markdown("<br>", unsafe_allow_html=True)

            if "CRITICAL" in status:
                st.error(f"🚨 **CRITICAL ALERT:** High failure probability detected! Torque ({torque} Nm) and Tool Wear ({tool_wear} min) have breached safe operational envelopes.")
            elif "WARNING" in status:
                st.warning(f"⚠️ **WARNING:** Machine operating under thermal/mechanical stress. Monitor temperature difference ({temp_diff:.1f}K) closely.")
            else:
                st.success("✅ **SYSTEM HEALTHY:** All telemetry inputs are within normal standard operating limits.")

            new_entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "User": st.session_state["username"],
                "Machine Type": machine_type,
                "Air Temp (K)": air_temp,
                "Process Temp (K)": process_temp,
                "RPM": rpm,
                "Torque (Nm)": torque,
                "Tool Wear (min)": tool_wear,
                "Failure Risk (%)": round(failure_prob, 2),
                "Health Score (%)": round(health_score, 2),
                "Status": status
            }
            st.session_state["prediction_history"] = pd.concat([pd.DataFrame([new_entry]), st.session_state["prediction_history"]], ignore_index=True)

    # ================= TAB 2: LIVE IOT STREAM =================
    with nav_tab2:
        st.markdown("""
        <div class="platform-header">
            <h1>AI-Based Predictive Maintenance and Machine Failure Prediction Using Sensor Data</h1>
            <p>Simulate real-time sensor data streaming from plant machinery connected via MQTT/gRPC.</p>
        </div>
        """, unsafe_allow_html=True)

        stream_col1, stream_col2 = st.columns([1, 2])
        with stream_col1:
            st.markdown("### ⚙️ Stream Control")
            selected_machine = st.selectbox("Select IoT Node", ["CNC-Machine-01", "Lathe-Unit-04", "Robotic-Arm-09"])
            stream_active = st.checkbox("🟢 Start Live Data Stream", value=False)
            
        with stream_col2:
            st.markdown(f"### 📈 Live Telemetry Monitor: `{selected_machine}`")
            placeholder_stream = st.empty()
            
        if stream_active:
            for _ in range(10):
                sim_torque = np.random.uniform(35.0, 65.0)
                sim_rpm = np.random.randint(1400, 1700)
                sim_temp = np.random.uniform(300.0, 320.0)
                
                with placeholder_stream.container():
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Live Torque", f"{sim_torque:.2f} Nm", f"{np.random.uniform(-1,1):.1f}")
                    s2.metric("Live RPM", f"{sim_rpm} RPM", f"{np.random.randint(-10,10)}")
                    s3.metric("Live Process Temp", f"{sim_temp:.1f} K", f"{np.random.uniform(-0.5,0.5):.1f}")
                time.sleep(1.0)
        else:
            placeholder_stream.info("ℹ️ Check 'Start Live Data Stream' above to initiate real-time telemetry simulation.")

    # ================= TAB 3: FLEET EXECUTIVE OVERVIEW =================
    with nav_tab3:
        st.markdown("""
        <div class="platform-header">
            <h1>AI-Based Predictive Maintenance and Machine Failure Prediction Using Sensor Data</h1>
            <p>Real-time fleet health monitoring and active machine node statuses.</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏭 Active Plant Nodes", "18 Units Online", "100% Operational")
        c2.metric("📊 Total Scans Run", f"{len(st.session_state['prediction_history'])} Records", "Active Session")
        c3.metric("🚨 System Status", "Stable", "No Immediate Outages")
        c4.metric("🛡️ Overall Fleet Health", "94.2%", "Optimal")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🌐 Plant Node Monitoring Table")
        nodes_df = pd.DataFrame([
            {"Node ID": "CNC-01", "Line": "Line A", "Status": "Optimal", "Health": "96.5%"},
            {"Node ID": "LATHE-04", "Line": "Line B", "Status": "Warning", "Health": "72.1%"},
            {"Node ID": "MILL-09", "Line": "Line C", "Status": "Optimal", "Health": "98.4%"},
            {"Node ID": "PRESS-12", "Line": "Line D", "Status": "Critical", "Health": "28.0%"}
        ])
        st.dataframe(nodes_df, use_container_width=True)

    # ================= TAB 4: TREND ANALYTICS =================
    with nav_tab4:
        st.markdown("""
        <div class="platform-header">
            <h1>AI-Based Predictive Maintenance and Machine Failure Prediction Using Sensor Data</h1>
            <p>Graphical breakdown of session prediction runs and failure risk timelines.</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state["prediction_history"].empty:
            df_h = st.session_state["prediction_history"]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 📉 Failure Risk Timeline")
                st.line_chart(df_h, x="Timestamp", y="Failure Risk (%)")
            with col_b:
                st.markdown("### 🛡️ Health Score Timeline")
                st.line_chart(df_h, x="Timestamp", y="Health Score (%)")
        else:
            st.info("ℹ️ No historical data available. Run a prediction in the **AI Prediction** tab first.")

    # ================= TAB 5: AI COPILOT =================
    with nav_tab5:
        st.markdown("""
        <div class="platform-header">
            <h1>AI-Based Predictive Maintenance and Machine Failure Prediction Using Sensor Data</h1>
            <p>Ask technical questions regarding machine anomalies and maintenance guidelines.</p>
        </div>
        """, unsafe_allow_html=True)

        for msg in st.session_state["copilot_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("Ask about torque limits, tool wear, or troubleshooting...")
        if query:
            st.session_state["copilot_messages"].append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            q_lower = query.lower()
            if "torque" in q_lower:
                reply = "Torque values above 70 Nm under high rotational speeds can accelerate bearing wear. Recommended maintenance interval is 48 hours."
            elif "tool wear" in q_lower:
                reply = "Standard tooling replacement threshold is 220 minutes. Exceeding 250 minutes drastically increases failure probability."
            else:
                reply = f"Continuous monitoring of air and process temperature differentials prevents unexpected thermal breakdowns. Let me know if you need specific parameter analysis!"

            st.session_state["copilot_messages"].append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

    # ================= TAB 6: AUDIT LOGS =================
    with nav_tab6:
        st.markdown("""
        <div class="platform-header">
            <h1>AI-Based Predictive Maintenance and Machine Failure Prediction Using Sensor Data</h1>
            <p>Complete traceability record of all diagnostic scans and predictions.</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state["prediction_history"].empty:
            st.dataframe(st.session_state["prediction_history"], use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            with d1:
                csv_data = st.session_state["prediction_history"].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Audit Logs (CSV)",
                    data=csv_data,
                    file_name=f"maintenance_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with d2:
                if st.button("🗑️ Clear History Logs", use_container_width=True, type="secondary"):
                    st.session_state["prediction_history"] = pd.DataFrame(columns=st.session_state["prediction_history"].columns)
                    st.success("Audit logs cleared successfully.")
                    st.rerun()
        else:
            st.info("ℹ️ Audit log registry is currently empty.")

    # ================= TAB 7: SETTINGS =================
    with nav_tab7:
        st.markdown("""
        <div class="platform-header">
            <h1>AI-Based Predictive Maintenance and Machine Failure Prediction Using Sensor Data</h1>
            <p>Manage user roles and system properties.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 👥 Authorized Roles")
        roles_df = pd.DataFrame([
            {"Username": "admin", "Role": "Super Administrator", "Access": "Full Control"},
            {"Username": "engineer", "Role": "Reliability Engineer", "Access": "Diagnostics & Logs"}
        ])
        st.dataframe(roles_df, use_container_width=True)

# ==============================================================================
# 7. ROUTER CONTROLLER
# ==============================================================================
if __name__ == '__main__':
    if not st.session_state["logged_in"]:
        render_login()
    else:
        render_dashboard()