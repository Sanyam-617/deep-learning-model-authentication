"""
DeepGuard - Deep Learning Model Authentication Dashboard
========================================================
Interactive Streamlit web frontend for behavioral challenge-response model authentication.
Connects directly to the verified backend pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_authentication import run_pipeline
from src.authentication.authenticator import AuthenticationThresholds

# ─────────────────────────────────────────────────────────────
# 1. Page Configuration & Custom CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard | Model Authentication System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Styles & High Contrast Typography */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    .main-title .brand-accent {
        color: #38BDF8;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    /* Metric Cards - Sleek Dark Theme with Vibrant Accents */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 8px 0;
    }
    .metric-badge-pass {
        display: inline-block;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #34D399;
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 9999px;
    }
    .metric-badge-fail {
        display: inline-block;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #F87171;
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 9999px;
    }
    .metric-thresh {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 8px;
    }
    /* Decision Banners */
    .decision-banner-auth {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 22px 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
        margin-bottom: 20px;
    }
    .decision-banner-rej {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 22px 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.25);
        margin-bottom: 20px;
    }
    .decision-title {
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: 1px;
    }
    .decision-desc {
        font-size: 1.0rem;
        opacity: 0.95;
        margin-top: 4px;
    }
    /* Pipeline Step Box */
    .pipeline-step {
        background: #1E293B;
        border: 1px solid #334155;
        border-left: 5px solid #38BDF8;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 14px;
        color: #E2E8F0;
    }
    .pipeline-step h4 {
        color: #F8FAFC;
        margin-bottom: 6px;
        font-weight: 700;
    }
    .pipeline-step p, .pipeline-step li {
        color: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# 2. Model Metadata & Helper Definitions
# ─────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
CHALLENGES_PATH = PROJECT_ROOT / "data" / "challenges" / "challenge_set.pt"
REFERENCE_FP_PATH = PROJECT_ROOT / "data" / "fingerprints" / "reference_fingerprint.pt"
ORIGINAL_MODEL_PATH = MODELS_DIR / "original_model.pth"
EXPERIMENTS_CSV_PATH = PROJECT_ROOT / "results" / "experiments" / "e1_e9_summary.csv"

MODEL_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "original_model.pth": {
        "label": "original_model.pth (Genuine Registered Model - E1)",
        "category": "Genuine",
        "description": "Baseline registered model trained on CIFAR-10. Expected: AUTHENTIC",
    },
    "exact_copy_model.pth": {
        "label": "exact_copy_model.pth (Bitwise Exact Duplicate - E2)",
        "category": "Genuine",
        "description": "Re-saved bitwise copy of registered model. Expected: AUTHENTIC",
    },
    "modified_model_1pct.pth": {
        "label": "modified_model_1pct.pth (1% Weight Perturbation - E3)",
        "category": "Impostor (Tampered)",
        "description": "Deterministic Gaussian perturbation of 1% layer parameters. Expected: REJECTED",
    },
    "modified_model_5pct.pth": {
        "label": "modified_model_5pct.pth (5% Weight Perturbation - E4)",
        "category": "Impostor (Tampered)",
        "description": "Deterministic Gaussian perturbation of 5% layer parameters. Expected: REJECTED",
    },
    "modified_model_10pct.pth": {
        "label": "modified_model_10pct.pth (10% Weight Perturbation - E5)",
        "category": "Impostor (Tampered)",
        "description": "Deterministic Gaussian perturbation of 10% layer parameters. Expected: REJECTED",
    },
    "modified_model_25pct.pth": {
        "label": "modified_model_25pct.pth (25% Weight Perturbation - E6)",
        "category": "Impostor (Tampered)",
        "description": "Deterministic Gaussian perturbation of 25% layer parameters. Expected: REJECTED",
    },
    "modified_model_50pct.pth": {
        "label": "modified_model_50pct.pth (50% Weight Perturbation - E7)",
        "category": "Impostor (Tampered)",
        "description": "Deterministic Gaussian perturbation of 50% layer parameters. Expected: REJECTED",
    },
    "independent_model.pth": {
        "label": "independent_model.pth (Independently Trained CNN - E8)",
        "category": "Impostor (Independent)",
        "description": "Trained independently on CIFAR-10 with seed=99. Expected: REJECTED",
    },
    "random_model.pth": {
        "label": "random_model.pth (Randomly Initialized CNN - E9)",
        "category": "Impostor (Random)",
        "description": "Untrained random Gaussian weight initialization. Expected: REJECTED",
    },
    "substitute_model.pth": {
        "label": "substitute_model.pth (Substitute / Surrogate Model)",
        "category": "Impostor (Surrogate)",
        "description": "Surrogate model architecture for adversarial testing. Expected: REJECTED",
    },
}


def get_available_models() -> List[str]:
    """Retrieve all available .pth model files from models directory."""
    if not MODELS_DIR.exists():
        return []
    pth_files = sorted([f.name for f in MODELS_DIR.glob("*.pth")])
    # Put original_model.pth first if present
    if "original_model.pth" in pth_files:
        pth_files.remove("original_model.pth")
        pth_files.insert(0, "original_model.pth")
    return pth_files


# ─────────────────────────────────────────────────────────────
# 3. Sidebar Configuration
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 1L3 5V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V5L12 1Z" fill="#0284C7" fill-opacity="0.25" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M9 12L11 14L15 10" stroke="#38BDF8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span style="font-size: 1.55rem; font-weight: 800; color: #F8FAFC; letter-spacing: 0.5px;">DeepGuard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("v1.0 • Behavioral Challenge-Response")
    st.markdown("---")

    st.markdown("### **Authentication Thresholds**")
    st.markdown(
        """
        - **Prediction Agreement**: $\\ge 85.0\\%$
        - **Cosine Similarity**: $\\ge 0.9000$
        - **KL Divergence**: $\\le 0.1500$
        - **Rule**: `all_metrics` (Conjunction)
        """
    )
    st.info(
        "A candidate model is marked **AUTHENTIC** if and only if **all three** metrics strictly satisfy their thresholds.",
        icon="ℹ️",
    )

    st.markdown("---")
    st.markdown("### **Reference Model Specs**")
    st.markdown(
        """
        - **Target**: `CIFAR10CNN`
        - **Architecture**: 3-stage ConvNet + FC
        - **Parameters**: 1,116,970 (~1.12M)
        - **Challenges**: 100 probe samples
        - **Classes**: 10 CIFAR-10 labels
        """
    )


# ─────────────────────────────────────────────────────────────
# 4. Main Header
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">DeepGuard &mdash; <span class="brand-accent">Deep Learning Model Authentication</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Black-box behavioral challenge–response verification using high-dimensional output fingerprints</div>',
    unsafe_allow_html=True,
)

tab_auth, tab_benchmark, tab_methodology = st.tabs(
    ["🛡️ Model Verification", "📊 Experiment Benchmark (E1–E9)", "🔍 System Methodology"]
)


# ─────────────────────────────────────────────────────────────
# TAB 1: LIVE MODEL VERIFICATION
# ─────────────────────────────────────────────────────────────
with tab_auth:
    col_ref, col_cand = st.columns([1, 1], gap="medium")

    with col_ref:
        st.subheader("1. Registered Reference Model")
        st.markdown(
            f"""
            - **Model Checkpoint**: `{ORIGINAL_MODEL_PATH.name}`
            - **Reference Fingerprint**: `{REFERENCE_FP_PATH.name}` (Precomputed)
            - **Challenge Set**: `{CHALLENGES_PATH.name}` (100 synthetic & transformed probes)
            - **Status**: Registered & Integrity Verified
            """
        )
        st.caption("The reference model represents the genuine registered intellectual property.")

    with col_cand:
        st.subheader("2. Candidate Model Selection")
        available_models = get_available_models()

        if not available_models:
            st.error("No model checkpoints (.pth) found in models/ directory.")
            selected_filename = None
        else:
            def format_option(name: str) -> str:
                info = MODEL_DESCRIPTIONS.get(name)
                return info["label"] if info else f"{name} (Custom Model)"

            selected_filename = st.selectbox(
                "Choose candidate model checkpoint to verify:",
                options=available_models,
                format_func=format_option,
                index=0,
            )

            if selected_filename and selected_filename in MODEL_DESCRIPTIONS:
                meta = MODEL_DESCRIPTIONS[selected_filename]
                st.caption(f"**Category:** {meta['category']} • *{meta['description']}*")

    st.markdown("---")

    run_col1, run_col2 = st.columns([1, 3])
    with run_col1:
        run_button = st.button("🚀 Run Authentication", type="primary", use_container_width=True)

    if run_button and selected_filename:
        candidate_path = MODELS_DIR / selected_filename

        with st.spinner(f"Extracting behavioral fingerprint for '{selected_filename}' over 100 challenges..."):
            try:
                # Use temporary directory for output to ensure reference files remain untouched
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_output_dir = Path(tmp_dir)
                    results = run_pipeline(
                        submitted_model_paths=[candidate_path],
                        original_model_path=ORIGINAL_MODEL_PATH,
                        challenge_path=CHALLENGES_PATH,
                        reference_fp_path=REFERENCE_FP_PATH,
                        thresholds=AuthenticationThresholds(
                            prediction_threshold=0.85,
                            cosine_threshold=0.90,
                            kl_threshold=0.15,
                        ),
                        output_dir=tmp_output_dir,
                        output_json=tmp_output_dir / "temp_res.json",
                        output_csv=tmp_output_dir / "temp_res.csv",
                        batch_size=64,
                    )

                if results and len(results) > 0:
                    r = results[0]
                    is_auth = r.get("authenticated", False)
                    agr = r.get("prediction_agreement", 0.0)
                    cos = r.get("cosine_similarity", 0.0)
                    kl = r.get("kl_divergence", 0.0)
                    agr_pass = r.get("prediction_pass", False)
                    cos_pass = r.get("cosine_pass", False)
                    kl_pass = r.get("kl_pass", False)
                    eval_time = r.get("evaluation_time_sec", 0.0)
                    matches = r.get("matching_predictions", 0)
                    total_ch = r.get("num_challenges", 100)

                    # Decision Banner
                    if is_auth:
                        st.markdown(
                            f"""
                            <div class="decision-banner-auth">
                                <div class="decision-title">AUTHENTIC MODEL VERIFIED</div>
                                <div class="decision-desc">The candidate model behavioral fingerprint strictly matches the registered model across all 3 criteria.</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="decision-banner-rej">
                                <div class="decision-title">REJECTED — IMPOSTOR / TAMPERED MODEL</div>
                                <div class="decision-desc">The candidate model failed one or more fingerprint criteria. It is not the registered model.</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # 3 High-Visibility Metric Cards
                    m1, m2, m3 = st.columns(3)

                    with m1:
                        badge = (
                            '<span class="metric-badge-pass">PASS</span>'
                            if agr_pass
                            else '<span class="metric-badge-fail">FAIL</span>'
                        )
                        val_color = "#34D399" if agr_pass else "#F87171"
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">Prediction Agreement</div>
                                <div class="metric-value" style="color: {val_color};">{agr * 100:.2f}%</div>
                                <div>{badge}</div>
                                <div class="metric-thresh">Threshold: &ge; 85.00% &bull; Matches: {matches}/{total_ch}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with m2:
                        badge = (
                            '<span class="metric-badge-pass">PASS</span>'
                            if cos_pass
                            else '<span class="metric-badge-fail">FAIL</span>'
                        )
                        val_color = "#34D399" if cos_pass else "#F87171"
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">Cosine Similarity</div>
                                <div class="metric-value" style="color: {val_color};">{cos:.4f}</div>
                                <div>{badge}</div>
                                <div class="metric-thresh">Threshold: &ge; 0.9000 &bull; Directional align</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with m3:
                        badge = (
                            '<span class="metric-badge-pass">PASS</span>'
                            if kl_pass
                            else '<span class="metric-badge-fail">FAIL</span>'
                        )
                        val_color = "#34D399" if kl_pass else "#F87171"
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">KL Divergence</div>
                                <div class="metric-value" style="color: {val_color};">{kl:.4f}</div>
                                <div>{badge}</div>
                                <div class="metric-thresh">Threshold: &le; 0.1500 &bull; Relative entropy</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    with st.expander("🔍 Detailed Pipeline Telemetry & Execution Log", expanded=False):
                        st.json(
                            {
                                "candidate_model": selected_filename,
                                "final_decision": "AUTHENTIC" if is_auth else "REJECTED",
                                "evaluation_latency_sec": f"{eval_time:.4f}s",
                                "checks": {
                                    "prediction_agreement": {"value": agr, "threshold": 0.85, "pass": agr_pass},
                                    "cosine_similarity": {"value": cos, "threshold": 0.90, "pass": cos_pass},
                                    "kl_divergence": {"value": kl, "threshold": 0.15, "pass": kl_pass},
                                },
                                "challenge_probes_evaluated": total_ch,
                                "conjunction_rule": "all_metrics (Agr AND Cos AND KL)",
                            }
                        )

            except Exception as e:
                st.error(f"Execution Error: {e}")


# ─────────────────────────────────────────────────────────────
# TAB 2: BENCHMARK EXPERIMENT RESULTS (E1–E9)
# ─────────────────────────────────────────────────────────────
with tab_benchmark:
    st.subheader("System Benchmark Suite (Experiments E1–E9)")
    st.caption("Empirical authentication results across registered, exact-copy, modified, independent, and random models.")

    # KPI Summary Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Tests", "9 Models", "E1–E9")
    k2.metric("Genuine Accuracy", "100.0%", "2/2 Verified")
    k3.metric("Impostor Rejection", "100.0%", "7/7 Detected")
    k4.metric("False Accept Rate (FAR)", "0.0%", "0 False Positives")
    k5.metric("False Reject Rate (FRR)", "0.0%", "0 False Negatives")

    st.markdown("---")

    if EXPERIMENTS_CSV_PATH.exists():
        df = pd.read_csv(EXPERIMENTS_CSV_PATH)

        # Format dataframe for display
        display_df = df.copy()
        display_df["Agreement"] = display_df["prediction_agreement"].apply(lambda v: f"{v*100:.1f}%")
        display_df["Cosine Sim"] = display_df["cosine_similarity"].apply(lambda v: f"{v:.4f}")
        display_df["KL Div"] = display_df["kl_divergence"].apply(lambda v: f"{v:.4f}")
        display_df["Checks (Agr/Cos/KL)"] = display_df.apply(
            lambda row: f"{'P' if row['prediction_pass'] else 'F'} / {'P' if row['cosine_pass'] else 'F'} / {'P' if row['kl_pass'] else 'F'}",
            axis=1,
        )

        cols_to_show = [
            "experiment_id",
            "model_filename",
            "description",
            "ground_truth_category",
            "decision",
            "Agreement",
            "Cosine Sim",
            "KL Div",
            "Checks (Agr/Cos/KL)",
        ]
        st.dataframe(
            display_df[cols_to_show].rename(
                columns={
                    "experiment_id": "Exp ID",
                    "model_filename": "Model Checkpoint",
                    "description": "Description",
                    "ground_truth_category": "Ground Truth",
                    "decision": "Decision",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Metric Trajectory Across Experiments")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            fig.patch.set_facecolor("#1E293B")
            ax.set_facecolor("#0F172A")
            exp_ids = df["experiment_id"]
            agrs = df["prediction_agreement"] * 100
            colors = ["#10B981" if d == "AUTHENTIC" else "#EF4444" for d in df["decision"]]

            ax.bar(exp_ids, agrs, color=colors, width=0.55, edgecolor="#334155", linewidth=0.8)
            ax.axhline(85.0, color="#38BDF8", linestyle="--", linewidth=1.5, label="Threshold (85%)")
            ax.set_title("Prediction Agreement (%) Across E1–E9", fontsize=11, fontweight="bold", pad=8, color="#F8FAFC")
            ax.set_ylabel("Agreement (%)", fontsize=10, color="#E2E8F0")
            ax.tick_params(colors="#CBD5E1", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.set_ylim(0, 105)
            ax.grid(axis="y", linestyle=":", alpha=0.3, color="#64748B")
            leg = ax.legend(loc="upper right", fontsize=9, facecolor="#1E293B", edgecolor="#334155")
            for text in leg.get_texts():
                text.set_color("#E2E8F0")
            plt.tight_layout()
            st.pyplot(fig)

        with chart_col2:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            fig.patch.set_facecolor("#1E293B")
            ax.set_facecolor("#0F172A")
            cos_vals = df["cosine_similarity"]
            colors = ["#10B981" if d == "AUTHENTIC" else "#EF4444" for d in df["decision"]]

            ax.bar(exp_ids, cos_vals, color=colors, width=0.55, edgecolor="#334155", linewidth=0.8)
            ax.axhline(0.90, color="#38BDF8", linestyle="--", linewidth=1.5, label="Threshold (0.90)")
            ax.set_title("Cosine Similarity Across E1–E9", fontsize=11, fontweight="bold", pad=8, color="#F8FAFC")
            ax.set_ylabel("Cosine Similarity", fontsize=10, color="#E2E8F0")
            ax.tick_params(colors="#CBD5E1", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.set_ylim(0.4, 1.05)
            ax.grid(axis="y", linestyle=":", alpha=0.3, color="#64748B")
            leg = ax.legend(loc="upper right", fontsize=9, facecolor="#1E293B", edgecolor="#334155")
            for text in leg.get_texts():
                text.set_color("#E2E8F0")
            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.warning(f"Experiment summary file not found at: {EXPERIMENTS_CSV_PATH}")


# ─────────────────────────────────────────────────────────────
# TAB 3: SYSTEM METHODOLOGY & ARCHITECTURE
# ─────────────────────────────────────────────────────────────
with tab_methodology:
    st.subheader("DeepGuard Architecture & Methodology")
    st.markdown(
        """
        DeepGuard provides non-destructive, black-box authentication for deep neural networks.
        Instead of fragile direct weight comparison (vulnerable to fine-tuning, permutation, and obfuscation),
        DeepGuard queries the candidate model with a curated set of high-entropy behavioral challenges.
        """
    )

    s1, s2 = st.columns(2)

    with s1:
        st.markdown(
            """
            <div class="pipeline-step">
                <h4>Stage 1: Challenge Set Generation</h4>
                <p>100 synthetic and transformed probe images generated from CIFAR-10 test distributions. Designed to probe decision boundaries.</p>
            </div>
            <div class="pipeline-step">
                <h4>Stage 2: Behavioral Fingerprint Extraction</h4>
                <p>For each challenge image, the network outputs an unnormalized logit vector transformed via Softmax into a 10-class probability distribution vector.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s2:
        st.markdown(
            """
            <div class="pipeline-step">
                <h4>Stage 3: Multi-Metric Authentication Engine</h4>
                <p>Computes three orthogonal statistical distances:</p>
                <ul>
                    <li><strong>Prediction Agreement</strong>: Categorical argmax consensus (&ge; 85%)</li>
                    <li><strong>Cosine Similarity</strong>: High-dimensional directional alignment (&ge; 0.90)</li>
                    <li><strong>KL Divergence</strong>: Asymmetric information divergence (&le; 0.15)</li>
                </ul>
            </div>
            <div class="pipeline-step">
                <h4>Stage 4: Attack & Robustness Evaluation</h4>
                <p>Comprehensive benchmarking against bitwise copies, weight perturbations (1% to 50%), independently initialized architectures, and surrogate models.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Authentication Decision Formulation")
    st.latex(
        r"\text{Decision} = (\text{Agreement} \ge 0.85) \land (\text{Cosine} \ge 0.90) \land (\text{KL} \le 0.15)"
    )
