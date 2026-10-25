import os
import time
import json
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import preprocess_text, preprocess_pipeline_steps
from src.model import (
    load_artifacts,
    predict_single_article,
    batch_predict,
    CATEGORY_COLORS,
    UNCERTAIN_LABEL,
    DEFAULT_CONFIDENCE_THRESHOLD
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="News Category Classification System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load External Custom CSS Styling
CSS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "styles.css")
if os.path.exists(CSS_FILE):
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sample Presets for Interactive Demo
SAMPLE_ARTICLES = {
    "Market Rally: Central Banks Signal Rate Cuts Following Inflation Drop": (
        "Global stock markets rallied on Friday after central bank officials signaled potential interest rate cuts "
        "following sharp declines in headline inflation figures. European equities closed at record highs led by "
        "banking and energy conglomerates, while crude oil futures stabilized around $82 per barrel. Major investment "
        "banks raised their corporate earnings forecasts for the upcoming fiscal quarter, citing resilient consumer spending."
    ),
    "AI Innovation: Next-Gen Sparse Attention Model Cuts Memory by 70%": (
        "Researchers at an artificial intelligence laboratory unveiled a next-generation multimodal neural network architecture "
        "capable of running complex computer vision and natural language reasoning locally on edge hardware devices. "
        "The new transformer model leverages 4-bit quantization and dynamic sparse attention mechanisms, achieving a 70% reduction "
        "in GPU memory consumption while maintaining near-zero latency in benchmark tests."
    ),
    "Premier League: Manchester City Mounts Sensational 3-2 Comeback Over Arsenal": (
        "Manchester City delivered a sensational second-half comeback to defeat Arsenal 3-2 in a thrilling Premier League match "
        "at the Etihad Stadium. The decisive goal arrived in the 94th minute when Kevin De Bruyne curled a magnificent 25-yard "
        "free kick into the top corner. With this victory, City narrow the gap at the top of the championship table with six matches remaining."
    ),
    "Parliament Session: Emergency Debate Over Electoral Reforms and Fiscal Bills": (
        "Parliament convened for an intense emergency debate today over proposed legislative reforms to regional election laws and "
        "public infrastructure spending. Ministers from the ruling party defended the taxation amendments, arguing they are critical "
        "for long-term fiscal stability, while opposition leaders demanded an independent parliamentary inquiry into campaign finance disclosures."
    ),
    "Box Office & Festival: Christopher Nolan Historical Epic Sweeps Top Honors": (
        "The 77th annual international film festival concluded with a standing ovation as director Christopher Nolan received the top "
        "honor for his historical epic drama. The cinematic release swept four major accolades, including Best Director, Best Original Screenplay, "
        "and Best Cinematography, marking one of the most celebrated box office and critical triumphs of the festival season."
    )
}

# Session State Initialization for Predictor
if "article_input_area" not in st.session_state:
    st.session_state["article_input_area"] = ""
if "preset_choice_select" not in st.session_state:
    st.session_state["preset_choice_select"] = "(Custom Input)"
if "has_run_prediction" not in st.session_state:
    st.session_state["has_run_prediction"] = False

# Callbacks for robust synchronization
def on_preset_change():
    chosen = st.session_state.get("preset_choice_select")
    if chosen and chosen != "(Custom Input)" and chosen in SAMPLE_ARTICLES:
        st.session_state["article_input_area"] = SAMPLE_ARTICLES[chosen]
        st.session_state["has_run_prediction"] = True
    elif chosen == "(Custom Input)":
        pass

def on_reset_click():
    current_val = st.session_state.get("article_input_area", "").strip()
    preset_val = st.session_state.get("preset_choice_select", "(Custom Input)")
    if not current_val and preset_val == "(Custom Input)":
        st.session_state["reset_notification"] = ("info", "Article input is already empty.")
    else:
        st.session_state["reset_notification"] = ("success", "Input cleared successfully.")
    
    st.session_state["article_input_area"] = ""
    st.session_state["preset_choice_select"] = "(Custom Input)"
    st.session_state["has_run_prediction"] = False

def on_debug_preset_change():
    chosen = st.session_state.get("debug_preset_select")
    if chosen and chosen in SAMPLE_ARTICLES:
        st.session_state["debug_text_area"] = SAMPLE_ARTICLES[chosen]

if "debug_preset_select" not in st.session_state:
    st.session_state["debug_preset_select"] = list(SAMPLE_ARTICLES.keys())[0]
if "debug_text_area" not in st.session_state:
    st.session_state["debug_text_area"] = SAMPLE_ARTICLES[st.session_state["debug_preset_select"]]



# Cache and Load Artifacts
@st.cache_resource(show_spinner="Loading NLP Pipeline and Models...")
def get_cached_artifacts():
    return load_artifacts()

try:
    vectorizer, models_dict, metrics = get_cached_artifacts()
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    load_error = str(e)

# Sidebar Navigation and Configuration
with st.sidebar:
    # 1. Clean Branding Block
    st.markdown("""
    <div class="sidebar-brand-box">
        <div class="brand-icon-wrapper">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8V6Z"/></svg>
        </div>
        <div class="brand-text-wrapper">
            <div class="brand-title">News Classifier</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Navigation Section with Micro-Typography
    st.markdown('<div class="sidebar-section-header">NAVIGATION</div>', unsafe_allow_html=True)
    navigation = st.radio(
        "Navigation",
        options=[
            "Live Article Classifier",
            "Dataset & Model Analytics",
            "NLP Pipeline Debugger",
            "Architecture & Methodology"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    # 3. Model Settings Section with Micro-Typography
    st.markdown('<div class="sidebar-section-header">MODEL CONFIGURATION</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "Active Algorithm",
        options=["Support Vector Machine (Linear SVM)", "Multinomial Naive Bayes"],
        index=0,
        help="Select the trained machine learning algorithm for predictions."
    )
    confidence_threshold = st.slider(
        "Confidence Guardrail Threshold",
        min_value=0.20,
        max_value=0.80,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        format="%.2f",
        help="Articles with prediction confidence below this threshold are flagged as Uncertain / Other."
    )
    
    # 4. Pipeline Status Section with Micro-Typography & Compact Glowing Card
    st.markdown('<div class="sidebar-section-header">SYSTEM HEALTH</div>', unsafe_allow_html=True)
    if pipeline_ready:
        st.markdown("""
        <div class="sidebar-status-card">
            <div class="status-indicator-row">
                <div class="status-dot"></div>
                <span class="status-title">Pipeline Online</span>
            </div>
            <div class="status-details">
                5,000 TF-IDF Features • Dual Models Ready
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Pipeline Offline. Check error log.")

# Main Application Routing
if not pipeline_ready:
    st.error(f"Failed to initialize machine learning models: {load_error}")
    st.stop()

# ==========================================
# 1. LIVE ARTICLE CLASSIFIER
# ==========================================
if navigation == "Live Article Classifier":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-content">
            <div class="header-badge-row">
                <span class="header-badge">LIVE INFERENCE</span>
            </div>
            <div class="header-title-text">News Category Classification</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode Switcher Tabs (Single Article vs Bulk CSV Batch)
    tab_single, tab_batch = st.tabs(["Single Article Inference", "Bulk Batch CSV Processing"])
    
    with tab_single:
        col_input, col_output = st.columns([1, 1], gap="medium")
        
        with col_input:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;'>Input News Content</div>", unsafe_allow_html=True)
                
                # Preset selector
                st.selectbox(
                    "Select a pre-configured sample article:",
                    options=["(Custom Input)"] + list(SAMPLE_ARTICLES.keys()),
                    key="preset_choice_select",
                    on_change=on_preset_change
                )
                
                input_text = st.text_area(
                    "Article Text Content",
                    height=145,
                    placeholder="Paste headline and article text here (e.g., market reports, tech breakthroughs, sports recaps, political updates...)",
                    key="article_input_area"
                )
                
                col_btn1, col_btn2 = st.columns([1.1, 1])
                with col_btn1:
                    classify_clicked = st.button("Classify Article", use_container_width=True, type="primary")
                    if classify_clicked:
                        if input_text.strip():
                            st.session_state["has_run_prediction"] = True
                        else:
                            st.warning("Please enter or select news article content before classifying.")
                            st.session_state["has_run_prediction"] = False
                with col_btn2:
                    st.button("Reset Input", use_container_width=True, on_click=on_reset_click)
                
                if "reset_notification" in st.session_state and st.session_state["reset_notification"]:
                    msg_type, msg_text = st.session_state["reset_notification"]
                    if msg_type == "info":
                        st.info(msg_text)
                    else:
                        st.success(msg_text)
                    st.session_state["reset_notification"] = None
            
        with col_output:
            has_input = bool(input_text.strip())
            should_classify = st.session_state.get("has_run_prediction", False) and has_input
            
            if should_classify:
                res = predict_single_article(
                    input_text,
                    selected_model_name,
                    vectorizer,
                    models_dict,
                    confidence_threshold=confidence_threshold
                )
                
                if "error" in res:
                    st.warning(res["error"])
                else:
                    pred_cat = res["category"]
                    cat_color = res["color"]
                    conf = res["confidence"]
                    lat = res["latency_ms"]
                    
                    with st.container(border=True):
                        st.markdown(f"""
                        <div class="prediction-hero-compact">
                            <div class="prediction-hero-header">
                                <span style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight:700;">PREDICTED CATEGORY</span>
                                <span style="font-size: 0.74rem; font-family: 'JetBrains Mono', monospace; color: #2563eb; font-weight:600;">{lat} ms latency</span>
                            </div>
                            <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px; margin: 2px 0 8px 0;">
                                <div class="category-badge-compact" style="background: {cat_color}18; color: {cat_color}; border: 1.5px solid {cat_color}55;">
                                    <span>{pred_cat}</span>
                                </div>
                                <div style="font-size: 1.1rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #0f172a;">
                                    {conf}% <span style="font-size: 0.72rem; color: #64748b; font-weight: 500;">confidence</span>
                                </div>
                            </div>
                            <div class="confidence-bar-container">
                                <div class="confidence-bar-fill" style="width: {conf}%; background: linear-gradient(90deg, {cat_color}, #0284c7);"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if res.get("is_uncertain") and res.get("uncertain_reason"):
                            st.markdown(f"""
                            <div style="background: #fffbeb; border: 1px solid #fde68a; border-left: 3.5px solid #f59e0b; border-radius: 6px; padding: 8px 12px; margin: 6px 0 10px 0; font-size: 0.8rem; color: #92400e;">
                                <strong>Guardrail Notice:</strong> {res["uncertain_reason"]}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Interactive Probability Distribution Plot
                        prob_df = pd.DataFrame({
                            "Category": list(res["probabilities"].keys()),
                            "Probability (%)": list(res["probabilities"].values())
                        })
                        
                        fig = px.bar(
                            prob_df,
                            x="Probability (%)",
                            y="Category",
                            orientation="h",
                            text="Probability (%)",
                            color="Category",
                            color_discrete_map=CATEGORY_COLORS
                        )
                        fig.update_traces(
                            texttemplate='%{text:.1f}%',
                            textposition='outside',
                            marker_line_color='rgba(0,0,0,0.06)',
                            marker_line_width=1
                        )
                        fig.update_layout(
                            template="plotly_white",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="#f8fafc",
                            height=180,
                            margin=dict(l=5, r=35, t=5, b=5),
                            showlegend=False,
                            font=dict(color="#0f172a", family="Plus Jakarta Sans", size=11),
                            xaxis=dict(range=[0, 115], showgrid=True, gridcolor="#e2e8f0"),
                            yaxis=dict(autorange="reversed", showgrid=False)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Top TF-IDF Drivers
                        st.markdown("<p style='font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; margin: 6px 0 2px 0;'>Top Discriminative TF-IDF Terms</p>", unsafe_allow_html=True)
                        if res.get("top_keywords"):
                            chips_html = "".join([f'<div class="keyword-chip"><span>{kw["word"]}</span><span class="chip-weight">{kw["tfidf_score"]}</span></div>' for kw in res["top_keywords"]])
                            st.html(f"<div class='chips-wrapper'>{chips_html}</div>")
                        else:
                            st.markdown("<div style='font-size: 0.8rem; color: #94a3b8; font-style: italic; padding: 4px 0;'>No active vocabulary terms matched in master dictionary.</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="empty-state-card">
                    <div class="empty-state-icon">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8V6Z"/></svg>
                    </div>
                    <div class="empty-state-title">Awaiting Article Input</div>
                    <div class="empty-state-desc">
                        Select a sample preset or enter news text on the left and click <strong>Classify Article</strong> to generate live predictions.
                    </div>
                    <div class="empty-state-tags">
                        <span class="empty-tag">Business</span>
                        <span class="empty-tag">Technology</span>
                        <span class="empty-tag">Sports</span>
                        <span class="empty-tag">Politics</span>
                        <span class="empty-tag">Entertainment</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_batch:
        with st.container(border=True):
            st.markdown("<h3 style='margin-top:0; font-size:1.15rem; font-weight:700; color:#0f172a;'>Bulk CSV Classification Pipeline</h3>", unsafe_allow_html=True)
            st.write("Upload a CSV of articles and choose its text column. Articles are grouped locally using the model trained from news_dataset.csv.")
            
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], help="CSV should contain a text column with news articles.")
            
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
                    if len(batch_df) == 0:
                        st.warning("The uploaded CSV contains no data rows.")
                    else:
                        st.write(f"Loaded **{len(batch_df)}** records. Preview:")
                        st.dataframe(batch_df.head(4), use_container_width=True)
                        
                        text_cols = list(batch_df.columns)
                        if not text_cols:
                            st.error("The uploaded CSV does not contain any column headers.")
                        else:
                            target_col = st.selectbox("Select Text Column to Classify:", options=text_cols, index=0)
                            batch_signature = (file_hash, target_col, selected_model_name, confidence_threshold)

                            saved_batch = st.session_state.get("batch_results")
                            if not saved_batch or saved_batch[0] != batch_signature:
                                with st.spinner("Processing batch records through TF-IDF and Classifier..."):
                                    res_batch, stats = batch_predict(
                                        batch_df,
                                        target_col,
                                        selected_model_name,
                                        vectorizer,
                                        models_dict,
                                        confidence_threshold=confidence_threshold
                                    )
                                    st.session_state["batch_results"] = (batch_signature, res_batch, stats)
                            _, res_batch, stats = st.session_state["batch_results"]
                            st.success(f"Classified {stats['total_count']} articles in {stats['total_time_s']}s.")
                    category_options = ["All categories", *CATEGORY_COLORS.keys()]
                    selected_category = st.selectbox(
                        "Filter articles by category:",
                        options=category_options,
                        key="batch_category_filter"
                    )
                    if selected_category == "All categories":
                        filtered_batch = res_batch
                    else:
                        filtered_batch = res_batch[
                            res_batch["Predicted_Category"] == selected_category
                        ]

                    st.write(f"Showing **{len(filtered_batch)}** of **{len(res_batch)}** articles.")
                    st.dataframe(filtered_batch, use_container_width=True)

                    csv_data = filtered_batch.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Displayed Articles as CSV",
                        data=csv_data,
                        file_name="filtered_news_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error reading uploaded CSV file: {str(e)}")

# ==========================================
# 2. DATASET & MODEL PERFORMANCE ANALYTICS
# ==========================================
elif navigation == "Dataset & Model Analytics":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-content">
            <div class="header-badge-row">
                <span class="header-badge">BENCHMARK & CORPUS</span>
            </div>
            <div class="header-title-text">Model Performance & Dataset Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    nb_m = metrics.get("Multinomial Naive Bayes", {})
    svm_m = metrics.get("Support Vector Machine (Linear SVM)", {})
    dedup_count = metrics.get('dataset_info', {}).get('deduplicated_samples', metrics.get('total_samples', 2127))
    test_size_count = metrics.get('test_size', 426)
    
    # Executive KPI Summary Cards
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card" style="border-left: 4px solid #7c3aed;">
            <div class="kpi-label">Linear SVM Accuracy</div>
            <div class="kpi-value" style="color: #7c3aed;">{svm_m.get('accuracy', 0) * 100:.2f}%</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #2563eb;">
            <div class="kpi-label">Naive Bayes Accuracy</div>
            <div class="kpi-value" style="color: #2563eb;">{nb_m.get('accuracy', 0) * 100:.2f}%</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #059669;">
            <div class="kpi-label">Deduplicated Corpus</div>
            <div class="kpi-value" style="color: #059669;">{dedup_count:,}</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #d97706;">
            <div class="kpi-label">TF-IDF Vocabulary</div>
            <div class="kpi-value" style="color: #d97706;">{metrics.get('feature_count', 5000):,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_bench1, col_bench2 = st.columns([1, 1], gap="medium")
    
    with col_bench1:
        with st.container(border=True):
            st.markdown("<h3 style='margin-top:0; font-size:1.15rem; font-weight:700; color:#0f172a;'>Model Comparison Metrics</h3>", unsafe_allow_html=True)
            
            comp_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Linear SVM": [
                    svm_m.get("accuracy", 0) * 100,
                    svm_m.get("precision", 0) * 100,
                    svm_m.get("recall", 0) * 100,
                    svm_m.get("f1_score", 0) * 100
                ],
                "Naive Bayes": [
                    nb_m.get("accuracy", 0) * 100,
                    nb_m.get("precision", 0) * 100,
                    nb_m.get("recall", 0) * 100,
                    nb_m.get("f1_score", 0) * 100
                ]
            })
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name="Linear SVM",
                x=comp_df["Metric"],
                y=comp_df["Linear SVM"],
                marker_color="#7c3aed",
                text=[f"{v:.2f}%" for v in comp_df["Linear SVM"]],
                textposition="outside"
            ))
            fig_comp.add_trace(go.Bar(
                name="Multinomial Naive Bayes",
                x=comp_df["Metric"],
                y=comp_df["Naive Bayes"],
                marker_color="#2563eb",
                text=[f"{v:.2f}%" for v in comp_df["Naive Bayes"]],
                textposition="outside"
            ))
            fig_comp.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#f8fafc",
                barmode="group",
                height=320,
                font=dict(color="#0f172a", family="Plus Jakarta Sans"),
                yaxis=dict(range=[90, 103], showgrid=True, gridcolor="#e2e8f0"),
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
    with col_bench2:
        with st.container(border=True):
            st.markdown("<h3 style='margin-top:0; font-size:1.15rem; font-weight:700; color:#0f172a;'>Dataset Category Distribution</h3>", unsafe_allow_html=True)
            
            # Category balance visualization
            cats = metrics.get("categories", ["Business", "Entertainment", "Politics", "Sports", "Technology"])
            cat_counts = metrics.get("dataset_info", {}).get("category_distribution", {
                "Sports": 505,
                "Business": 503,
                "Politics": 403,
                "Entertainment": 369,
                "Technology": 347
            })
            
            fig_dist = px.bar(
                x=list(cat_counts.keys()),
                y=list(cat_counts.values()),
                color=list(cat_counts.keys()),
                color_discrete_map=CATEGORY_COLORS,
                text=list(cat_counts.values()),
                labels={"x": "News Category", "y": "Document Count"}
            )
            fig_dist.update_traces(textposition="outside")
            fig_dist.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#f8fafc",
                height=320,
                showlegend=False,
                font=dict(color="#0f172a", family="Plus Jakarta Sans"),
                yaxis=dict(range=[0, 600], showgrid=True, gridcolor="#e2e8f0"),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
    # Confusion Matrix Section
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0; font-size:1.15rem; font-weight:700; color:#0f172a;'>Interactive Confusion Matrix Heatmap</h3>", unsafe_allow_html=True)
        
        col_cm_ctrl, col_cm_plot = st.columns([1, 2.5])
        with col_cm_ctrl:
            cm_model_choice = st.radio(
                "Select Evaluation Model for Heatmap:",
                options=["Support Vector Machine (Linear SVM)", "Multinomial Naive Bayes"]
            )
            normalize_cm = st.checkbox("Normalize Heatmap by Row (% True Class)", value=False)
            
            st.markdown(f"""
            <div style="font-size: 0.82rem; color: #64748b; margin-top: 15px; line-height: 1.5;">
                The confusion matrix illustrates classification fidelity across the held-out 20% test partition ({test_size_count} articles). Diagonal cells indicate accurate predictions, while off-diagonals highlight minor cross-domain semantic overlap.
            </div>
            """, unsafe_allow_html=True)
            
        with col_cm_plot:
            active_cm = np.array(metrics.get(cm_model_choice, {}).get("confusion_matrix", []))
            if len(active_cm) > 0:
                if normalize_cm:
                    row_sums = active_cm.sum(axis=1, keepdims=True)
                    active_cm_display = np.round((active_cm / row_sums) * 100, 1)
                    text_template = "%{z:.1f}%"
                    zmin, zmax = 0, 100
                else:
                    active_cm_display = active_cm
                    text_template = "%{z} docs"
                    zmin, zmax = 0, active_cm.max()
                    
                fig_cm = px.imshow(
                    active_cm_display,
                    x=cats,
                    y=cats,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Class", y="Ground Truth Class", color="Count" if not normalize_cm else "%"),
                    text_auto=True,
                    zmin=zmin,
                    zmax=zmax
                )
                fig_cm.update_traces(texttemplate=text_template)
                fig_cm.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    font=dict(color="#0f172a", family="Plus Jakarta Sans"),
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_cm, use_container_width=True)
        
    # Detailed Per-Class Classification Report Table
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0; font-size:1.15rem; font-weight:700; color:#0f172a;'>Per-Class Classification Report (Linear SVM)</h3>", unsafe_allow_html=True)
        
        report_dict = svm_m.get("classification_report", {})
        if report_dict:
            report_rows = []
            for cat in cats:
                if cat in report_dict:
                    report_rows.append({
                        "Category": cat,
                        "Precision": f"{report_dict[cat]['precision'] * 100:.2f}%",
                        "Recall": f"{report_dict[cat]['recall'] * 100:.2f}%",
                        "F1-Score": f"{report_dict[cat]['f1-score'] * 100:.2f}%",
                        "Support (Test Docs)": int(report_dict[cat]['support'])
                    })
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

# ==========================================
# 3. INTERACTIVE NLP PIPELINE DEBUGGER
# ==========================================
elif navigation == "NLP Pipeline Debugger":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-content">
            <div class="header-badge-row">
                <span class="header-badge">NLP TRACER</span>
            </div>
            <div class="header-title-text">NLP Pipeline Inspector</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_deb_input, col_deb_stats = st.columns([2, 1], gap="medium")
    
    with col_deb_input:
        with st.container(border=True):
            st.markdown("<h3 style='margin-top:0; font-size:1.15rem; font-weight:700; color:#0f172a;'>Input Text for Real-Time Pipeline Tracing</h3>", unsafe_allow_html=True)
            
            st.selectbox(
                "Select sample to trace through NLP steps:",
                options=list(SAMPLE_ARTICLES.keys()),
                key="debug_preset_select",
                on_change=on_debug_preset_change
            )
            debug_text = st.text_area(
                "Trace Text",
                height=130,
                key="debug_text_area"
            )
        
    steps_data = preprocess_pipeline_steps(debug_text)
    stats = steps_data["stats"]
    
    with col_deb_stats:
        with st.container(border=True):
            st.markdown("<h3 style='margin-top:0; font-size:1.15rem; font-weight:700; color:#0f172a;'>Lexical Metrics</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; line-height: 2;">
                <div>Original Word Count: <span style="color: #2563eb; float: right; font-weight:700;">{stats['raw_word_count']}</span></div>
                <div>Raw NLTK Tokens: <span style="color: #7c3aed; float: right; font-weight:700;">{stats['token_count']}</span></div>
                <div>Sanitized Tokens: <span style="color: #0284c7; float: right; font-weight:700;">{stats['cleaned_word_count']}</span></div>
                <div>Stopwords Filtered: <span style="color: #dc2626; float: right; font-weight:700;">{stats['removed_stopword_count']}</span></div>
                <div>Terms Lemmatized: <span style="color: #059669; float: right; font-weight:700;">{stats.get('lemmatized_count', 0)}</span></div>
                <div>Retained Base Lemmas: <span style="color: #059669; float: right; font-weight:700;">{stats['retained_token_count']}</span></div>
                <div>Corpus Compression: <span style="color: #0284c7; float: right; font-weight:700;">{stats['reduction_percentage']}%</span></div>
                <div>Lexical Density: <span style="color: #d97706; float: right; font-weight:700;">{stats['vocabulary_density']}%</span></div>
            </div>
            """, unsafe_allow_html=True)
        
    # Visual Step-by-Step Breakdown
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0; font-size:1.15rem; font-weight:700; color:#0f172a;'>Transformation Stages</h3>", unsafe_allow_html=True)
        
        # Stage 1: Tokenization
        raw_tokens_chips = "".join([f'<span style="background: #f1f5f9; border: 1px solid #cbd5e1; color: #1e293b; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; font-family: monospace; display: inline-block; margin: 2px;">{t}</span>' for t in steps_data['raw_tokens']])
        stage1_html = (
            f'<div class="step-card">'
            f'<div class="step-header"><span>Stage 1: Word & Sentence Tokenization ({len(steps_data["raw_tokens"])} total tokens)</span><span class="step-badge">NLTK punkt Tokenizer</span></div>'
            f'<div style="font-size: 0.76rem; color: #64748b; margin-bottom: 6px;">Segments continuous raw text into distinct linguistic units, words, and punctuation markers:</div>'
            f'<div style="display: flex; flex-wrap: wrap; gap: 6px; max-height: 180px; overflow-y: auto;">{raw_tokens_chips}</div>'
            f'</div>'
        )
        st.html(stage1_html)
        
        # Stage 2: Normalization & Sanitization
        stage2_html = (
            f'<div class="step-card">'
            f'<div class="step-header"><span>Stage 2: Lowercase Normalization & Character Sanitization ({len(steps_data.get("cleaned_tokens", []))} valid tokens)</span><span class="step-badge">Regex Sanitized</span></div>'
            f'<div style="font-size: 0.76rem; color: #64748b; margin-bottom: 6px;">Converts text to lowercase, strips HTML/URLs, and removes punctuation, numbers, and special symbols:</div>'
            f'<div style="color: #1e293b; font-size: 0.88rem; font-family: \'JetBrains Mono\', monospace; background: #f8fafc; border: 1px solid #e2e8f0; padding: 12px; border-radius: 8px;">{steps_data["cleaned_text"]}</div>'
            f'</div>'
        )
        st.html(stage2_html)
        
        # Stage 3: Stopword Removal
        noise_html = ""
        if steps_data.get("short_noise"):
            noise_tokens = " ".join([f'<code>{n}</code>' for n in steps_data['short_noise']])
            noise_html = f'<div style="font-size: 0.72rem; color: #94a3b8; margin-top: 6px;">Short noise tokens filtered (&le; 2 chars): {noise_tokens}</div>'
        
        sw_chips = "".join([f'<span style="background: #fee2e2; color: #b91c1c; padding: 2px 7px; border-radius: 6px; font-size: 0.78rem; text-decoration: line-through; display: inline-block; margin: 2px;">{s}</span>' for s in steps_data['removed_stopwords']])
        stage3_html = (
            f'<div class="step-card">'
            f'<div class="step-header"><span>Stage 3: Stopword & Noise Elimination ({len(steps_data["removed_stopwords"])} unique stopwords filtered)</span><span class="step-badge" style="background: #fef2f2; color: #dc2626; border-color: #fecaca;">Filtered</span></div>'
            f'<div style="font-size: 0.76rem; color: #64748b; margin-bottom: 6px;">Removes frequent grammatical function words with low semantic discriminator value:</div>'
            f'<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 2px;">{sw_chips}</div>'
            f'{noise_html}'
            f'</div>'
        )
        st.html(stage3_html)
        
        # Stage 4: Morphological Lemmatization with Arrows
        transformations = steps_data.get("transformations", [])
        trans_chips = "".join([
            f'<span style="background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; font-family: monospace; display: inline-block; margin: 2px;"><span style="text-decoration: line-through; opacity: 0.6;">{tr["original"]}</span> &rarr; <strong>{tr["lemma"]}</strong></span>'
            for tr in transformations
        ])
        
        trans_block = ""
        if transformations:
            trans_block = f'<div style="margin-bottom: 12px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px 12px;"><div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #065f46; margin-bottom: 6px;">Morphological Reductions ({len(transformations)} inflected words brought to base lemma):</div><div style="display: flex; flex-wrap: wrap; gap: 6px; max-height: 150px; overflow-y: auto;">{trans_chips}</div></div>'
        
        stage4_html = (
            f'<div class="step-card">'
            f'<div class="step-header"><span>Stage 4: WordNet Morphological Lemmatization ({len(steps_data["lemmatized_tokens"])} retained base lemmas)</span><span class="step-badge" style="background: #ecfdf5; color: #059669; border-color: #a7f3d0;">Base Lemmas</span></div>'
            f'{trans_block}'
            f'<div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 5px;">Final Normalized Document (Ready for TF-IDF Vectorization):</div>'
            f'<div style="color: #047857; font-size: 0.88rem; font-family: \'JetBrains Mono\', monospace; background: #f8fafc; border: 1px solid #e2e8f0; padding: 12px; border-radius: 8px; line-height: 1.6;">{steps_data["final_output"]}</div>'
            f'</div>'
        )
        st.html(stage4_html)
        
        # Stage 5: TF-IDF Vector Extraction
        if steps_data['final_output']:
            tfidf_vec = vectorizer.transform([steps_data['final_output']])
            feature_names = np.array(vectorizer.get_feature_names_out())
            row = tfidf_vec.tocoo()
            sorted_items = sorted(zip(row.col, row.data), key=lambda x: x[1], reverse=True)
            tfidf_chips = "".join([f'<div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 6px 10px; border-radius: 8px; font-size: 0.8rem; font-family: monospace;"><span style="color: #0f172a; font-weight:600;">{feature_names[idx]}</span> <span style="color: #2563eb; float: right; font-weight:700;">{val:.4f}</span></div>' for idx, val in sorted_items])
            stage5_html = (
                f'<div class="step-card">'
                f'<div class="step-header"><span>Stage 5: High-Dimensional TF-IDF Vector Projection ({len(sorted_items)} active vocabulary matches)</span><span class="step-badge" style="background: #eff6ff; color: #2563eb; border-color: #bfdbfe;">5,000-D Master Vocabulary</span></div>'
                f'<div style="font-size: 0.8rem; color: #64748b; margin-bottom: 8px;">Matched words from the 5,000-term dictionary and their relative importance weights:</div>'
                f'<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; margin-top: 4px; max-height: 240px; overflow-y: auto;">{tfidf_chips}</div>'
                f'</div>'
            )
            st.html(stage5_html)

# ==========================================
# 4. PROJECT OVERVIEW & ARCHITECTURE
# ==========================================
elif navigation == "Architecture & Methodology":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-content">
            <div class="header-badge-row">
                <span class="header-badge">ENGINEERING DOCS</span>
            </div>
            <div class="header-title-text">Architecture & Methodology</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Clean Segmented Tabs for Perfect Layout Alignment
    tab_arch_flow, tab_arch_math, tab_arch_tax = st.tabs([
        "Pipeline Architecture",
        "Model Comparison",
        "Dataset Taxonomy & Specifications"
    ])
    
    with tab_arch_flow:
        col_flow1, col_flow2 = st.columns([1.1, 1], gap="medium")
        
        with col_flow1:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;'>End-to-End NLP & Inference Workflow</div>", unsafe_allow_html=True)
                st.markdown("""
                ```mermaid
                graph TD
                    A["Raw News Article Input"] --> B["Text Preprocessing"]
                    B --> B1["Regex Sanitization"]
                    B1 --> B2["NLTK Word Tokenization"]
                    B2 --> B3["Stopword Filtration"]
                    B3 --> B4["WordNet Lemmatization"]
                    B4 --> C["TF-IDF Vectorizer (5,000 Features)"]
                    C --> D["Linear SVM / Naive Bayes"]
                    D --> E["Probabilistic Category Inferences"]
                    D --> F["Top Discriminative Feature Attribution"]
                ```
                """, unsafe_allow_html=False)
                
        with col_flow2:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;'>Pipeline Engineering Highlights</div>", unsafe_allow_html=True)
                
                st.markdown("""
                - **Sublinear TF Scaling:** Logarithmic frequency transformation prevents disproportionate weighting of recurring tokens.
                - **Unigram & Bigram Features:** Extracts single words and composite pairs (e.g., `interest rate`, `premier league`) for rich context.
                - **Calibrated Decision Boundaries:** Platt's sigmoid scaling maps SVM hyperplane margins to calibrated probabilistic distributions.
                - **WordNet Morphological Rooting:** Lemmatizes inflected words to unified semantic bases without stemming distortions.
                """)
                
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; margin-top: 14px;">
                    <div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 4px;">SYSTEM METRICS</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #0f172a; line-height: 1.8;">
                        • Vocabulary Dimension: <span style="color:#2563eb; font-weight:700;">5,000 terms</span><br>
                        • Corpus Split: <span style="color:#059669; font-weight:700;">2,127 Deduplicated Articles (80% Train / 20% Test)</span><br>
                        • Average Latency: <span style="color:#7c3aed; font-weight:700;">&lt; 5 ms / article</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_arch_math:
        st.markdown("<div style='font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-top: 4px; margin-bottom: 14px;'>How Each Model Thinks (Core Intuition)</div>", unsafe_allow_html=True)
        col_math1, col_math2 = st.columns([1, 1], gap="medium")
        
        with col_math1:
            with st.container(border=True):
                st.markdown("""
                <div class="algo-header">
                    <div>
                        <div class="algo-title">1. Linear Support Vector Machine</div>
                        <div class="algo-subtitle">Discriminative • Wide-Margin Boundary Finder • Production Engine</div>
                    </div>
                    <span class="algo-badge-blue">98.36% Accuracy</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="algo-section-heading">Core Intuition</div>
                <div class="algo-intuition-box">
                    <strong>"Finding the Widest Highway:"</strong> Imagine every news article is plotted on a map with 5,000 dimensions. Rather than counting word frequencies, SVM searches for a clean dividing line (hyperplane) with the widest possible empty corridor (margin) between categories.
                </div>
                
                <div class="algo-section-heading">Decision Flow (3 Steps)</div>
                <div class="algo-step-container">
                    <div class="algo-step-row">
                        <span class="algo-step-pill">1</span>
                        <div><strong>TF-IDF Vector Projection:</strong> Converts vocabulary words and unigram/bigram phrases into numerical weights.</div>
                    </div>
                    <div class="algo-step-row">
                        <span class="algo-step-pill">2</span>
                        <div><strong>Boundary Margin Scoring:</strong> Computes the signed geometric distance from the boundary line for all 5 categories.</div>
                    </div>
                    <div class="algo-step-row">
                        <span class="algo-step-pill">3</span>
                        <div><strong>Support Vector Anchor:</strong> The model ignores easy center articles and focuses on the hardest borderline articles to optimize precision.</div>
                    </div>
                </div>
                
                <div class="algo-section-heading">Production Deployment Rationale</div>
                <div style="font-size: 0.96rem; color: #334155; line-height: 1.7;">
                    • <strong>Word Relationships:</strong> Captures compound terms (like <em>"interest rate"</em> or <em>"prime minister"</em>) without assuming words are independent.<br>
                    • <strong>Boundary Precision:</strong> Yields the highest overall accuracy (<strong>98.36%</strong>) and lowest cross-domain confusion.
                </div>
                """, unsafe_allow_html=True)
        
        with col_math2:
            with st.container(border=True):
                st.markdown("""
                <div class="algo-header">
                    <div>
                        <div class="algo-title">2. Multinomial Naive Bayes</div>
                        <div class="algo-subtitle">Generative • Probabilistic Word Counter • High-Speed Baseline</div>
                    </div>
                    <span class="algo-badge-emerald">97.65% Accuracy</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="algo-section-heading">Core Intuition</div>
                <div class="algo-intuition-box algo-intuition-box-emerald">
                    <strong>"Multiplying Word Odds:"</strong> Naive Bayes looks at historical statistics to see how frequently each word appears in each topic. It calculates: <em>"Given all the words in this article, which category is statistically most likely to have generated them?"</em>
                </div>
                
                <div class="algo-section-heading">Decision Flow (3 Steps)</div>
                <div class="algo-step-container">
                    <div class="algo-step-row">
                        <span class="algo-step-pill algo-step-pill-emerald">1</span>
                        <div><strong>Word Frequency Tally:</strong> Counts occurrences of each vocabulary word present in the preprocessed article.</div>
                    </div>
                    <div class="algo-step-row">
                        <span class="algo-step-pill algo-step-pill-emerald">2</span>
                        <div><strong>Laplace Smoothing (+1 Safety Net):</strong> If an article contains a rare or unseen word, it adds a tiny pseudo-count so probability never collapses to 0%.</div>
                    </div>
                    <div class="algo-step-row">
                        <span class="algo-step-pill algo-step-pill-emerald">3</span>
                        <div><strong>Log-Odds Summation:</strong> Adds log-probabilities together to prevent tiny floating-point decimals from rounding down to zero.</div>
                    </div>
                </div>
                
                <div class="algo-section-heading">Benchmark Baseline Characteristics</div>
                <div style="font-size: 0.96rem; color: #334155; line-height: 1.7;">
                    • <strong>Instant Training:</strong> Calculates parameters in a single pass of counting (&lt; 0.2s) with zero iterative gradient steps.<br>
                    • <strong>Ultra-Low Latency:</strong> Executes inferences in under 1 ms, making it ideal for massive stream filtering.
                </div>
                """, unsafe_allow_html=True)
        
        # 3. Comprehensive Comparison Matrix
        with st.container(border=True):
            st.markdown("<div style='font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 12px;'>Executive Comparison Matrix</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="font-size: 0.96rem; line-height: 1.7; color: #1e293b;">
            """, unsafe_allow_html=True)
            
            st.markdown(r"""
            | Evaluation Factor | Support Vector Machine (Linear SVM) | Multinomial Naive Bayes (MNB) | Takeaway |
            | :--- | :--- | :--- | :--- |
            | **Model Paradigm** | **Discriminative** (draws boundary lines) | **Generative** (calculates word probabilities) | SVM directly separates classes; MNB models word generation. |
            | **Decision Rule** | Distance from hyperplane ($w^T x + b$) | Sum of log word probabilities | Both yield calibrated percentage confidence scores. |
            | **BBC News Accuracy** | **98.36%** (+0.71% edge) | **97.65%** | SVM wins by handling subtle word co-occurrences. |
            | **Handling Unseen Words** | Words outside vocabulary get zero weight | **Laplace Smoothing ($\alpha=0.1$)** prevents 0% collapse | Laplace smoothing guarantees mathematical stability. |
            | **Training Speed** | ~1.2s (iterative coordinate solver) | **~0.15s** (instant counting) | MNB trains near-instantaneously; SVM trains fast. |
            | **Inference Latency** | **< 2 ms** / article | **< 1 ms** / article | Both operate in real-time with sub-millisecond speeds. |
            | **Production Role** | **Selected Primary Production Model** | **High-Efficiency Benchmark Model** | Linear SVM chosen for higher boundary accuracy. |
            """)
            
            st.markdown("</div>", unsafe_allow_html=True)


    with tab_arch_tax:
        with st.container(border=True):
            st.markdown("<div style='font-size: 0.98rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>BBC News Corpus Category Taxonomy (2,127 Deduplicated Documents)</div>", unsafe_allow_html=True)
            
            st.markdown("""
            | Category | Documents | Domain Focus | Sample Key Discriminative Drivers |
            | :--- | :---: | :--- | :--- |
            | **Sports** | 505 | Football, Athletics, Rugby, Tennis, Championships | *game, win, team, player, match, victory, cup, coach, score* |
            | **Business** | 503 | Economy, Markets, Corporate, Banking, Trade | *growth, bank, market, share, economy, profit, trade, firm* |
            | **Politics** | 403 | Government, Parliament, Legislation, Elections | *minister, party, government, election, prime, vote, labor* |
            | **Entertainment** | 369 | Cinema, Music, Television, Awards, Arts | *film, music, award, star, show, actor, band, director, album* |
            | **Technology** | 347 | Software, Hardware, AI, Telecom, Cybersecurity | *technology, user, software, computer, mobile, broadband, net* |
            """)
            
            st.markdown("""
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 10px; line-height: 1.5;">
                <strong>Dataset Provenance:</strong> The BBC News raw corpus consists of 2,225 topic-annotated articles spanning 2004–2005. After dropping 98 exact duplicates to prevent train/test leakage, 2,127 verified unique documents were evaluated under a stratified 80/20 train/test evaluation protocol (1,701 train / 426 test).
            </div>
            """, unsafe_allow_html=True)
