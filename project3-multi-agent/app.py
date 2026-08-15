# app.py - Project 3 Streamlit UI
# Multi-Agent DB Monitor Web Interface

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import streamlit as st
import tempfile
from agents import scanner_agent, analyst_agent, reporter_agent

# ── Page config ────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent DB Monitor",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────
st.markdown("""
<style>
.header {
    background: linear-gradient(135deg, #0D1B2A, #6C3483);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
.agent-card {
    border-radius: 8px;
    padding: 12px;
    margin: 5px 0;
}
.score-big {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1 style="color:white; margin:0;">🤖 Multi-Agent Oracle DB Monitor</h1>
    <p style="color:#AED6F1; margin:5px 0 0 0;">
        3 AI Agents Collaborating | Scanner → Analyst → Reporter
    </p>
    <p style="color:#AED6F1; margin:3px 0 0 0;">
        Powered by RAG + Llama 3 + LangChain
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    db_name = st.text_input("Database Name", value="PRODDB")
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. 🔍 **Scanner** reads your log")
    st.markdown("2. 🧠 **Analyst** diagnoses errors")
    st.markdown("3. 📄 **Reporter** generates report")
    st.divider()
    st.markdown("**3 Agents. 0 Human steps.**")
    st.markdown("**Built by:** Srikanth")
    st.markdown("**Stack:** LangChain + RAG + Llama 3")

# ── Main layout ────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Paste Oracle Alert Log")
    log_content = st.text_area(
        "Alert Log",
        height=300,
        placeholder="""Paste your Oracle alert log here...

Example:
Fri Jan 15 14:30:22 2024
ORA-04031: unable to allocate memory
Fri Jan 15 14:35:10 2024
ORA-01555: snapshot too old
Fri Jan 15 14:45:30 2024
ORA-00060: deadlock detected""",
        label_visibility="collapsed"
    )

    # Load sample button
    if st.button("📂 Load Sample Log", type="secondary",
                  use_container_width=True):
        st.session_state.use_sample = True
        st.rerun()

    if st.session_state.get("use_sample"):
        log_content = """Fri Jan 15 14:30:22 2024
ORA-04031: unable to allocate 4096 bytes of shared memory
Fri Jan 15 14:31:05 2024
ORA-04031: unable to allocate 2048 bytes of shared memory
Fri Jan 15 14:35:10 2024
ORA-01555: snapshot too old: rollback segment number 5
Fri Jan 15 14:40:05 2024
ORA-00060: deadlock detected while waiting for resource
Fri Jan 15 14:50:12 2024
ORA-12541: TNS:no listener
Fri Jan 15 14:55:00 2024
ORA-01555: snapshot too old: rollback segment number 3
Fri Jan 15 15:00:00 2024
ORA-04031: unable to allocate 8192 bytes of shared memory"""

    # Analyze button
    analyze = st.button(
        "🚀 Run All 3 Agents",
        type="primary",
        use_container_width=True
    )

with col2:
    st.subheader("📊 Analysis Results")

    if analyze and log_content:

        # Write log to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.log',
            delete=False
        ) as tmp:
            tmp.write(log_content)
            tmp_path = tmp.name

        # ── AGENT 1: SCANNER ──────────────────────
        st.markdown("### 🔍 Agent 1 — Scanner")
        with st.spinner("Scanning alert log..."):
            scan_result = scanner_agent(tmp_path)

        if "error" in scan_result:
            st.error(f"Scanner failed: {scan_result['error']}")
            st.stop()

        # Show scanner results
        s1, s2, s3 = st.columns(3)
        s1.metric("Total Errors",
                   scan_result["total_errors"])
        s2.metric("Unique Types",
                   scan_result["unique_errors"])
        s3.metric("Scanned At",
                   scan_result["scanned_at"].split(" ")[1]
                   if " " in str(scan_result["scanned_at"])
                   else "Done")

        # Error breakdown
        with st.expander("View Error Breakdown", expanded=True):
            for code, count in sorted(
                scan_result["error_counts"].items(),
                key=lambda x: -x[1]
            ):
                st.markdown(f"**{code}** — {count} occurrence(s)")

        st.success("✅ Agent 1 Scanner complete!")
        st.divider()

        # ── AGENT 2: ANALYST ──────────────────────
        st.markdown("### 🧠 Agent 2 — Analyst")
        st.info("Diagnosing errors using RAG knowledge base + Llama 3...")

        with st.spinner(
            "Analysing... (30-60 sec for multiple errors)"
        ):
            analysis_result = analyst_agent(scan_result)

        if "error" in analysis_result:
            st.error(f"Analyst failed: {analysis_result['error']}")
            st.stop()

        # Health score display
        score = analysis_result["health_score"]
        grade = analysis_result["grade"]

        if score >= 75:
            score_color = "green"
        elif score >= 50:
            score_color = "orange"
        else:
            score_color = "red"

        st.markdown(
            f'<div class="score-big" style="color:{score_color}">'
            f'{score}/100</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h3 style='text-align:center;color:{score_color}'>"
            f"{grade}</h3>",
            unsafe_allow_html=True
        )

        # Diagnosis per error
        for code, data in analysis_result["diagnoses"].items():
            with st.expander(
                f"🔍 {code} — {data['count']} occurrence(s)"
            ):
                st.markdown(data["diagnosis"])

        st.success("✅ Agent 2 Analyst complete!")
        st.divider()

        # ── AGENT 3: REPORTER ─────────────────────
        st.markdown("### 📄 Agent 3 — Reporter")
        with st.spinner("Generating incident report..."):
            report_result = reporter_agent(
                analysis_result, db_name
            )

        st.success(
            f"✅ Report saved: {report_result['filename']}"
        )

        # Download button
        st.download_button(
            label="📥 Download Incident Report",
            data=report_result["report"],
            file_name=report_result["filename"],
            mime="text/plain",
            use_container_width=True
        )

        # Final summary
        st.divider()
        st.markdown("### 🎯 Pipeline Complete!")
        f1, f2 = st.columns(2)
        f1.metric("Final Score",
                   f"{report_result['health_score']}/100")
        f2.metric("Grade", report_result["grade"].split("—")[0])
        st.balloons()

    elif analyze and not log_content:
        st.warning("Please paste alert log content first!")

    else:
        st.markdown("""
        <div style='text-align:center;padding:60px;color:#7F8C8D;'>
            <h2>🤖 3 Agents Ready</h2>
            <p>Paste your Oracle alert log</p>
            <p>and click Run All 3 Agents</p>
            <br/>
            <p>Scanner → Analyst → Reporter</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Multi-Agent Oracle DB Monitor | "
    "LangChain + RAG + Llama 3 + Streamlit | "
    "github.com/geekycloud2026/agentic-ai-journey"
    "</small></center>",
    unsafe_allow_html=True
)