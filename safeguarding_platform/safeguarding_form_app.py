import streamlit as st
from datetime import datetime, date
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertModel, pipeline
import json
import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv(r"C:\Users\Alex\Documents\dissertation-ai\safeguarding_platform\.env")

MODEL_DIR = r"C:\Users\Alex\Documents\dissertation-ai\safeguarding_platform\models\multitask_v3"
DATABASE_URL = os.environ.get("DATABASE_URL")
LOGO_PATH = r"C:\Users\Alex\Documents\dissertation-ai\safeguarding_platform\acf_logo.png"

with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    model_config = json.load(f)

URGENCY_LABELS = model_config["urgency_labels"]
CATEGORY_LABELS = model_config["category_labels"]
label2id = model_config["label2id"]
id2label = {int(k): v for k, v in model_config["id2label"].items()}
category2id = model_config["category2id"]
id2category = {int(k): v for k, v in model_config["id2category"].items()}
MAX_LENGTH = model_config["max_length"]

REPORTER_CATEGORIES = [
    "Bullying",
    "Physical safety",
    "Mental health",
    "Home issues",
    "Online safety",
    "Financial hardship",
    "Attendance / engagement",
    "Behaviour / conduct",
    "Abuse by adult in organisation",
    "Abuse by another young person",
    "Sexual abuse / assault",
    "Grooming",
    "Radicalisation / extremism",
    "Exploitation / trafficking",
    "FGM / harmful practices",
    "Other",
]

CASEWORKER_CATEGORIES = ["-- Accept AI suggestion --"] + CATEGORY_LABELS + ["Other"]
CASEWORKER_URGENCY = ["-- Accept AI suggestion --"] + URGENCY_LABELS


# ---------------------------------------------------------
# Branding and custom CSS
# ---------------------------------------------------------

def apply_branding():
    st.markdown("""
    <style>
        :root {
            --acf-dark-green: #004D40;
            --acf-green: #00695C;
            --acf-light-green: #E0F2F1;
            --acf-red: #B71C1C;
            --acf-amber: #E65100;
            --acf-yellow: #F57F17;
            --acf-text: #212121;
        }
        section[data-testid="stSidebar"] {
            background-color: #004D40 !important;
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: white !important;
        }
        header[data-testid="stHeader"] {
            background-color: #004D40 !important;
        }
        h1 { color: #004D40 !important; }
        h2, h3 { color: #00695C !important; }
        div[data-testid="stMetric"] {
            background-color: #E0F2F1;
            border-left: 4px solid #004D40;
            padding: 10px 15px;
            border-radius: 4px;
        }
        div[data-testid="stMetric"] label {
            color: #004D40 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #004D40 !important;
        }
        .stFormSubmitButton button {
            background-color: #004D40 !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 2rem !important;
            font-weight: 600 !important;
        }
        .stFormSubmitButton button:hover {
            background-color: #00695C !important;
        }
        .stButton button {
            background-color: #004D40 !important;
            color: white !important;
            border: none !important;
        }
        .stButton button:hover {
            background-color: #00695C !important;
        }
        .streamlit-expanderHeader {
            font-weight: 600 !important;
        }
        .narrative-box {
            background-color: #FAFAFA;
            border-left: 4px solid #004D40;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 10px;
            font-size: 0.95em;
            line-height: 1.6;
        }
        .urgency-critical {
            background-color: #B71C1C;
            color: white;
            padding: 3px 10px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .urgency-high {
            background-color: #E65100;
            color: white;
            padding: 3px 10px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .urgency-medium {
            background-color: #F57F17;
            color: white;
            padding: 3px 10px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .urgency-low {
            background-color: #2E7D32;
            color: white;
            padding: 3px 10px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .success-box {
            background-color: #E0F2F1;
            border-left: 4px solid #004D40;
            padding: 20px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .mismatch-warning {
            background-color: #FFF3E0;
            border-left: 4px solid #E65100;
            padding: 10px 15px;
            border-radius: 4px;
            margin: 5px 0;
            font-size: 0.9em;
        }
        .person-link-alert {
            background-color: #FFEBEE;
            border-left: 4px solid #B71C1C;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 3px 0;
            font-size: 0.9em;
        }
        .footer-text {
            text-align: center;
            color: #9E9E9E;
            font-size: 0.8em;
            padding: 20px 0;
            border-top: 1px solid #E0E0E0;
            margin-top: 40px;
        }
    </style>
    """, unsafe_allow_html=True)


def urgency_badge(urgency):
    css_class = f"urgency-{urgency.lower()}"
    return f'<span class="{css_class}">{urgency}</span>'


# ---------------------------------------------------------
# Multi-task model definition
# ---------------------------------------------------------

class MultiTaskSafeguardingModel(nn.Module):
    def __init__(self, model_name, num_urgency_labels, num_category_labels):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.distilbert.config.hidden_size
        self.urgency_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_urgency_labels),
        )
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_category_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.urgency_head(cls_output), self.category_head(cls_output)


# ---------------------------------------------------------
# Load models (cached)
# ---------------------------------------------------------

@st.cache_resource
def load_classification_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = MultiTaskSafeguardingModel(
        model_name="distilbert-base-uncased",
        num_urgency_labels=len(URGENCY_LABELS),
        num_category_labels=len(CATEGORY_LABELS),
    )
    state_dict = torch.load(os.path.join(MODEL_DIR, "model_state.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def load_ner_model():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")


# ---------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------

def predict(text):
    tokenizer, model, device = load_classification_model()
    encoded = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        urg_logits, cat_logits = model(encoded["input_ids"], encoded["attention_mask"])
        urg_probs = F.softmax(urg_logits, dim=-1).squeeze()
        cat_probs = F.softmax(cat_logits, dim=-1).squeeze()
    pred_urgency = id2label[torch.argmax(urg_probs).item()]
    pred_category = id2category[torch.argmax(cat_probs).item()]
    secondary_categories = []
    for i, prob in enumerate(cat_probs.tolist()):
        cat_name = id2category[i]
        if cat_name != pred_category and prob >= 0.15:
            secondary_categories.append({"label": cat_name, "confidence": prob})
    secondary_categories.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "urgency_label": pred_urgency,
        "urgency_confidence": torch.max(urg_probs).item(),
        "category_label": pred_category,
        "category_confidence": torch.max(cat_probs).item(),
        "secondary_categories": secondary_categories,
    }


def extract_names(text):
    ner = load_ner_model()
    results = ner(text)
    seen_names = {}
    for entity in results:
        if entity["entity_group"] == "PER" and entity["score"] > 0.7:
            name = entity["word"].strip()
            if len(name) > 1:
                name_lower = name.lower()
                if name_lower not in seen_names or entity["score"] > seen_names[name_lower]["confidence"]:
                    seen_names[name_lower] = {"name": name, "confidence": float(entity["score"])}
    return list(seen_names.values())


# ---------------------------------------------------------
# Policy override logic
# ---------------------------------------------------------

def apply_policy_overrides(text, staff_category, base_urgency):
    t = text.lower().strip()
    order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    inv_order = {v: k for k, v in order.items()}
    current_level = order.get(base_urgency, 1)
    critical_keywords = ["kill myself", "end my life", "don't want to be alive", "suicide", "suicidal", "overdose", "hang myself", "jump off", "plan to hurt myself"]
    physical_abuse_keywords = ["hit me", "hit them", "punched", "kicked", "beat me", "grabbed me by the neck", "threw me"]
    sexual_abuse_keywords = ["touched me inappropriately", "touched them inappropriately", "private area", "warned me not to tell", "told me not to tell anyone", "raped"]
    exploitation_keywords = ["much older individuals", "much older men", "gives me money", "pick me up late at night"]
    min_level = current_level
    if any(kw in t for kw in critical_keywords):
        min_level = max(min_level, order["Critical"])
    if any(kw in t for kw in physical_abuse_keywords):
        min_level = max(min_level, order["High"])
    if any(kw in t for kw in sexual_abuse_keywords):
        if "abuse" in staff_category.lower() or "adult" in staff_category.lower() or "sexual" in staff_category.lower():
            min_level = max(min_level, order["Critical"])
        else:
            min_level = max(min_level, order["High"])
    if any(kw in t for kw in exploitation_keywords):
        min_level = max(min_level, order["High"])
    high_min_categories = ["Abuse by adult in organisation", "Exploitation / trafficking", "FGM / harmful practices", "Sexual abuse / assault"]
    if staff_category in high_min_categories and min_level < order["High"]:
        min_level = order["High"]
    return inv_order[min_level]


# ---------------------------------------------------------
# Database functions
# ---------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def save_report(data):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reports (unit_id, reporter_role, location, age_band, channel, staff_category, free_text,
            model_urgency, model_urgency_confidence, model_category, model_category_confidence,
            policy_urgency, category_mismatch, reporter_name, reporter_email, reporter_phone)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (data["unit_id"], data["reporter_role"], data["location"], data["age_band"], data["channel"],
         data["staff_category"], data["free_text"], data["model_urgency"], data["model_urgency_confidence"],
         data["model_category"], data["model_category_confidence"], data["policy_urgency"], data["category_mismatch"],
         data["reporter_name"], data["reporter_email"], data["reporter_phone"]))
    report_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return report_id

def save_extracted_persons(report_id, persons):
    if not persons:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    for person in persons:
        cur.execute("INSERT INTO extracted_persons (report_id, person_name, confidence) VALUES (%s, %s, %s)",
                    (report_id, person["name"], person["confidence"]))
    conn.commit()
    cur.close()
    conn.close()

def save_caseworker_review(report_id, urgency, category, notes, reviewer, status):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE reports SET caseworker_urgency = %s, caseworker_category = %s, caseworker_notes = %s,
            reviewed_by = %s, reviewed_at = NOW(), status = %s WHERE id = %s""",
        (urgency, category, notes, reviewer, status, report_id))
    conn.commit()
    cur.close()
    conn.close()

def find_name_matches(report_id, persons):
    if not persons:
        return []
    conn = get_db_connection()
    cur = conn.cursor()
    matches = []
    for person in persons:
        cur.execute("""
            SELECT ep.person_name, ep.report_id, r.submitted_at, r.staff_category, r.policy_urgency, r.free_text
            FROM extracted_persons ep JOIN reports r ON ep.report_id = r.id
            WHERE LOWER(ep.person_name) = LOWER(%s) AND ep.report_id != %s ORDER BY r.submitted_at DESC""",
            (person["name"], report_id))
        for row in cur.fetchall():
            matches.append({"matched_name": row[0], "report_id": row[1], "submitted_at": row[2],
                            "category": row[3], "urgency": row[4],
                            "narrative_preview": row[5][:150] + "..." if len(row[5]) > 150 else row[5]})
    cur.close()
    conn.close()
    return matches

def get_all_reports():
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT *,
            CASE COALESCE(caseworker_urgency, policy_urgency)
                WHEN 'Critical' THEN 1
                WHEN 'High' THEN 2
                WHEN 'Medium' THEN 3
                WHEN 'Low' THEN 4
                ELSE 5
            END as urgency_sort
        FROM reports
        ORDER BY urgency_sort ASC, submitted_at DESC
    """, conn)
    conn.close()
    return df

def get_persons_for_report(report_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT person_name, confidence FROM extracted_persons WHERE report_id = %s", (report_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"name": row[0], "confidence": row[1]} for row in rows]

def get_all_name_links():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT LOWER(person_name) as name, COUNT(DISTINCT report_id) as report_count,
               ARRAY_AGG(DISTINCT report_id) as report_ids
        FROM extracted_persons GROUP BY LOWER(person_name)
        HAVING COUNT(DISTINCT report_id) > 1 ORDER BY report_count DESC""")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"name": row[0], "report_count": row[1], "report_ids": row[2]} for row in rows]


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(page_title="ACF Safeguarding Platform", page_icon="\U0001F6E1", layout="wide")
apply_branding()

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    st.markdown("---")
    st.markdown("### Safeguarding Platform")
    page = st.radio("Navigate", ["Submit a Concern", "Caseworker Dashboard", "Trend Analysis", "Person Links"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<p style='font-size: 0.75em; color: rgba(255,255,255,0.6);'>AI-assisted decision support tool<br>Prototype for research purposes only</p>", unsafe_allow_html=True)


# =========================================================
# PAGE 1: Submit a Concern
# =========================================================

if page == "Submit a Concern":
    st.title("Report a Safeguarding Concern")
    st.markdown("Use this form to record a welfare or safeguarding concern about a young person. Your report will be reviewed by the safeguarding team.")

    with st.form("concern_form"):
        st.subheader("Concern Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            unit_id = st.text_input("Unit", value="")
        with col2:
            reporter_role = st.selectbox("Your role", ["Adult Volunteer (CFAV)", "Service Personnel", "Parent / Carer", "Family Member", "Cadet", "Other"], index=None, placeholder="Select...")
        with col3:
            location = st.selectbox("Where did this occur?", ["Parade Night", "Weekend Exercise", "Annual Camp", "Field Exercise", "Online / Remote", "Off-site Activity", "Home", "Other"], index=None, placeholder="Select...")

        col4, col5, col6 = st.columns(3)
        with col4:
            age_band = st.selectbox("Age band of young person", ["N/A", "Under 12", "12-14", "14-16", "16-18"], index=None, placeholder="Select...")
        with col5:
            channel = st.selectbox("How was this reported to you?", ["In person", "Online form", "Phone call", "Email", "Third party", "Other"], index=None, placeholder="Select...")
        with col6:
            incident_date = st.date_input("Date of incident", format="DD/MM/YYYY")

        st.markdown("---")
        st.subheader("Your Details")

        det_col1, det_col2, det_col3 = st.columns(3)
        with det_col1:
            reporter_name = st.text_input("Your name")
        with det_col2:
            reporter_email = st.text_input("Contact email")
        with det_col3:
            reporter_phone = st.text_input("Contact phone number")

        st.markdown("---")
        st.subheader("Concern Information")

        staff_category = st.selectbox("What type of concern is this?", REPORTER_CATEGORIES, index=None, placeholder="Select...")

        free_text = st.text_area("Describe what has been observed or disclosed", height=200,
            placeholder="Include the full names of all individuals involved where known (e.g. the young person, any adults, other cadets). "
            "Describe what you saw or were told, including any specific language used. State the specific location where this occurred if applicable.")

        submitted = st.form_submit_button("Submit Concern")

    if submitted:
        if not free_text.strip():
            st.warning("Please describe the concern before submitting.")
        else:
            prediction = predict(free_text)
            policy_urgency = apply_policy_overrides(free_text, staff_category, prediction["urgency_label"])
            staff_cat_lower = staff_category.lower().strip()
            model_cat_lower = prediction["category_label"].lower().strip()
            category_mismatch = staff_cat_lower not in model_cat_lower and model_cat_lower not in staff_cat_lower
            extracted_persons = extract_names(free_text)
            report_data = {
                "unit_id": unit_id, "reporter_role": reporter_role, "location": location,
                "age_band": age_band, "channel": channel, "staff_category": staff_category,
                "free_text": free_text, "model_urgency": prediction["urgency_label"],
                "model_urgency_confidence": prediction["urgency_confidence"],
                "model_category": prediction["category_label"],
                "model_category_confidence": prediction["category_confidence"],
                "policy_urgency": policy_urgency, "category_mismatch": category_mismatch,
                "reporter_name": reporter_name, "reporter_email": reporter_email,
                "reporter_phone": reporter_phone,
            }
            try:
                report_id = save_report(report_data)
                save_extracted_persons(report_id, extracted_persons)
                st.markdown(
                    f"<div class='success-box'>"
                    f"<h3 style='color: #004D40; margin-top: 0;'>Concern Received</h3>"
                    f"<p>Your concern has been received and will be reviewed by the safeguarding team.</p>"
                    f"<p style='font-size: 1.2em;'><strong>Reference number: {report_id}</strong></p>"
                    f"<p>Please keep this reference number for your records. If the situation changes or you have further information, "
                    f"please submit a new report and quote this reference.</p></div>",
                    unsafe_allow_html=True)
            except Exception as e:
                st.error("There was a problem saving your report. Please try again or contact the safeguarding team directly.")
                st.exception(e)


# =========================================================
# PAGE 2: Caseworker Dashboard
# =========================================================

elif page == "Caseworker Dashboard":
    st.title("Caseworker Dashboard")
    st.markdown("Overview of submitted safeguarding concerns with AI-assisted triage.")

    df = get_all_reports()

    if df.empty:
        st.info("No reports have been submitted yet.")
    else:
        # Summary metrics
        st.subheader("Overview")
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        with mc1:
            st.metric("Total Reports", len(df))
        with mc2:
            df["display_urgency"] = df.apply(lambda r: r["caseworker_urgency"] if pd.notna(r.get("caseworker_urgency")) else r["policy_urgency"], axis=1)
            st.metric("Critical", len(df[df["display_urgency"] == "Critical"]))
        with mc3:
            st.metric("High", len(df[df["display_urgency"] == "High"]))
        with mc4:
            st.metric("Medium", len(df[df["display_urgency"] == "Medium"]))
        with mc5:
            st.metric("Low", len(df[df["display_urgency"] == "Low"]))
        with mc6:
            reviewed = len(df[df["reviewed_at"].notna()])
            st.metric("Reviewed", f"{reviewed}/{len(df)}")

        st.markdown("---")

        # Filters
        st.subheader("Filters")
        fc1, fc2 = st.columns(2)
        with fc1:
            urgency_filter = st.multiselect("Urgency", options=["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
        with fc2:
            status_filter = st.selectbox("Status", options=["All", "Open", "In progress", "Closed"], index=1)

        filtered = df[df["display_urgency"].isin(urgency_filter)]
        if status_filter != "All":
            filtered = filtered[filtered["status"] == status_filter]

        st.markdown(f"Showing **{len(filtered)}** of **{len(df)}** reports")
        st.markdown("---")

        # Report cards
        for _, row in filtered.iterrows():
            urgency_icons = {"Critical": "\U0001F534", "High": "\U0001F7E0", "Medium": "\U0001F7E1", "Low": "\U0001F7E2"}
            display_urgency = row["caseworker_urgency"] if pd.notna(row.get("reviewed_at")) and row.get("caseworker_urgency") else row["policy_urgency"]
            icon = urgency_icons.get(display_urgency, "")

            # Status tag
            if row.get("status") == "Closed":
                status_tag = " \u2705"
            elif row.get("status") == "In progress":
                status_tag = " \u2014 Pending"
            else:
                status_tag = " \u2014 To review"

            display_category = row["caseworker_category"] if pd.notna(row.get("caseworker_category")) else row["model_category"]
            with st.expander(
                f"{icon} Report #{row['id']} | {display_urgency} | {display_category} | "
                f"{row['submitted_at'].strftime('%d %b %Y %H:%M') if pd.notna(row['submitted_at']) else 'Unknown'}"
                f"{status_tag}"
            ):
                detail_left, detail_right = st.columns([2, 1])

                with detail_left:
                    st.markdown("**Concern Narrative**")
                    st.markdown(f"<div class='narrative-box'>{row['free_text']}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"**Reporter:** {row['reporter_role']} &nbsp;|&nbsp; "
                        f"**Unit:** {row['unit_id']} &nbsp;|&nbsp; "
                        f"**Location:** {row['location']} &nbsp;|&nbsp; "
                        f"**Age band:** {row['age_band']} &nbsp;|&nbsp; "
                        f"**Channel:** {row['channel']}", unsafe_allow_html=True)
                    if row.get("reporter_name"):
                        contact_parts = [f"**Contact:** {row['reporter_name']}"]
                        if row.get("reporter_email"):
                            contact_parts.append(row["reporter_email"])
                        if row.get("reporter_phone"):
                            contact_parts.append(row["reporter_phone"])
                        st.markdown(" &nbsp;|&nbsp; ".join(contact_parts), unsafe_allow_html=True)

                with detail_right:
                    st.markdown("**AI Assessment**")

                    # Show caseworker override if reviewed, otherwise AI
                    if pd.notna(row.get("reviewed_at")) and row.get("caseworker_urgency"):
                        st.markdown(
                            f"Urgency: {urgency_badge(row['caseworker_urgency'])} &nbsp; <em>set by caseworker</em>",
                            unsafe_allow_html=True)
                        st.markdown(
                            f"<span style='color: #757575; font-size: 0.85em;'>"
                            f"AI suggested: {row['model_urgency']} ({row['model_urgency_confidence']:.0%})</span>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"Urgency: {urgency_badge(row['model_urgency'])} &nbsp; {row['model_urgency_confidence']:.0%} confident",
                            unsafe_allow_html=True)
                        if row["policy_urgency"] != row["model_urgency"]:
                            st.markdown(
                                f"Policy-adjusted: {urgency_badge(row['policy_urgency'])} &nbsp; <em>elevated by safety rules</em>",
                                unsafe_allow_html=True)

                    st.markdown("---")

                    # Category display
                    if pd.notna(row.get("reviewed_at")) and row.get("caseworker_category"):
                        st.markdown(
                            f"**Category:** {row['caseworker_category']} <em>(set by caseworker)</em>",
                            unsafe_allow_html=True)
                        st.markdown(
                            f"<span style='color: #757575; font-size: 0.85em;'>"
                            f"AI suggested: {row['model_category']} ({row['model_category_confidence']:.0%})</span>",
                            unsafe_allow_html=True)
                    else:
                        if row['model_category_confidence'] >= 0.30:
                            st.markdown(f"**AI category:** {row['model_category']} ({row['model_category_confidence']:.0%})")
                        else:
                            st.markdown(f"**AI category:** Unable to determine ({row['model_category_confidence']:.0%} confidence) — caseworker assessment required")
                        secondary = predict(row["free_text"]).get("secondary_categories", [])
                        if secondary:
                            sec_text = ", ".join(f"{s['label']} ({s['confidence']:.0%})" for s in secondary)
                            st.markdown(f"**Also flagged:** {sec_text}")

                    st.markdown(f"**Reporter category:** {row['staff_category']}")

                    if row["category_mismatch"] and pd.isna(row.get("reviewed_at")):
                        st.markdown(
                            "<div class='mismatch-warning'>\u26A0\uFE0F <strong>Category mismatch.</strong> "
                            "The AI suggests a different category to the reporter.</div>",
                            unsafe_allow_html=True)

                    st.markdown("---")

                    # Extracted persons
                    persons = get_persons_for_report(row["id"])
                    if persons:
                        st.markdown("**Persons Identified**")
                        for p in persons:
                            other_mentions = find_name_matches(row["id"], [p])
                            if other_mentions:
                                st.markdown(
                                    f"<div class='person-link-alert'>\U0001F517 <strong>{p['name']}</strong> "
                                    f"({p['confidence']:.0%}) \u2014 mentioned in {len(other_mentions)} other report(s)</div>",
                                    unsafe_allow_html=True)
                            else:
                                st.markdown(f"\U0001F464 {p['name']} ({p['confidence']:.0%})")

                # ---- Caseworker review section ----
                st.markdown("---")
                rid = row["id"]

                if row.get("status") == "Closed" and pd.notna(row.get("reviewed_at")):
                    # Closed case - show review summary, no editing
                    st.markdown("**Caseworker Review** \u2705 Closed")
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown(f"Urgency: **{row['caseworker_urgency']}**")
                        st.markdown(f"Category: **{row['caseworker_category']}**")
                    with rc2:
                        st.markdown(f"Reviewed by: **{row['reviewed_by']}**")
                        st.markdown(f"Reviewed: **{row['reviewed_at'].strftime('%d %b %Y %H:%M') if pd.notna(row['reviewed_at']) else ''}**")
                    if row.get("caseworker_notes"):
                        st.markdown(f"**Notes:** {row['caseworker_notes']}")
                    urg_agree = row["caseworker_urgency"] == row["model_urgency"]
                    cat_agree = row["caseworker_category"] == row["model_category"]
                    if urg_agree and cat_agree:
                        st.success("Caseworker agreed with both AI urgency and category.")
                    elif not urg_agree and not cat_agree:
                        st.info(f"Caseworker adjusted urgency ({row['model_urgency']} \u2192 {row['caseworker_urgency']}) and category ({row['model_category']} \u2192 {row['caseworker_category']}).")
                    elif not urg_agree:
                        st.info(f"Caseworker adjusted urgency: {row['model_urgency']} \u2192 {row['caseworker_urgency']}.")
                    else:
                        st.info(f"Caseworker adjusted category: {row['model_category']} \u2192 {row['caseworker_category']}.")

                elif pd.notna(row.get("reviewed_at")):
                    # In progress - show saved review AND allow editing
                    st.markdown("**Caseworker Review** \u2014 Pending")
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown(f"Current urgency: **{row['caseworker_urgency']}**")
                        st.markdown(f"Current category: **{row['caseworker_category']}**")
                    with rc2:
                        st.markdown(f"Reviewed by: **{row['reviewed_by']}**")
                        st.markdown(f"Last updated: **{row['reviewed_at'].strftime('%d %b %Y %H:%M') if pd.notna(row['reviewed_at']) else ''}**")
                    if row.get("caseworker_notes"):
                        st.markdown(f"**Notes:** {row['caseworker_notes']}")

                    st.markdown("---")
                    st.markdown("**Update review:**")
                    uc1, uc2 = st.columns(2)
                    with uc1:
                        upd_urgency = st.selectbox("Update urgency", options=CASEWORKER_URGENCY, key=f"upd_urg_{rid}")
                        upd_category = st.selectbox("Update category", options=CASEWORKER_CATEGORIES, key=f"upd_cat_{rid}")
                    with uc2:
                        upd_reviewer = st.text_input("Your name", value=row.get("reviewed_by", "") or "", key=f"upd_name_{rid}")
                        upd_status = st.selectbox("Update status", options=["In progress", "Closed"], key=f"upd_status_{rid}")
                    upd_notes = st.text_area("Update notes", value=row.get("caseworker_notes", "") or "", key=f"upd_notes_{rid}", height=80)
                    if st.button("Update Review", key=f"upd_save_{rid}"):
                        final_urg = row["caseworker_urgency"] if upd_urgency == "-- Accept AI suggestion --" else upd_urgency
                        final_cat = row["caseworker_category"] if upd_category == "-- Accept AI suggestion --" else upd_category
                        save_caseworker_review(rid, final_urg, final_cat, upd_notes, upd_reviewer, upd_status)
                        st.success(f"Review updated for Report #{rid}.")
                        st.rerun()

                else:
                    # Not yet reviewed - show fresh review form
                    st.markdown("**Caseworker Review**")
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        cw_urgency = st.selectbox("Your urgency assessment", options=CASEWORKER_URGENCY, key=f"cw_urg_{rid}")
                        cw_category = st.selectbox("Your category assessment", options=CASEWORKER_CATEGORIES, key=f"cw_cat_{rid}")
                    with rc2:
                        cw_reviewer = st.text_input("Your name", key=f"cw_name_{rid}")
                        cw_status = st.selectbox("Update status", options=["Open", "In progress", "Closed"], key=f"cw_status_{rid}")
                    cw_notes = st.text_area("Caseworker notes (optional)", key=f"cw_notes_{rid}", height=80,
                        placeholder="Record your assessment, actions taken, or rationale for any changes.")
                    if st.button("Save Review", key=f"cw_save_{rid}"):
                        if not cw_reviewer.strip():
                            st.warning("Please enter your name before saving the review.")
                        else:
                            final_urg = row["model_urgency"] if cw_urgency == "-- Accept AI suggestion --" else cw_urgency
                            final_cat = row["model_category"] if cw_category == "-- Accept AI suggestion --" else cw_category
                            save_caseworker_review(rid, final_urg, final_cat, cw_notes, cw_reviewer, cw_status)
                            st.success(f"Review saved for Report #{rid}.")
                            st.rerun()

        st.markdown("---")

# =========================================================
# PAGE 3: Trend Analysis
# =========================================================

elif page == "Trend Analysis":
    st.title("Trend Analysis")
    st.markdown("Strategic overview of safeguarding concern patterns and trends.")

    df = get_all_reports()

    if len(df) < 2:
        st.info("More reports are needed to show trend analysis.")
    else:
        df["final_urgency"] = df.apply(
            lambda r: r["caseworker_urgency"] if pd.notna(r.get("caseworker_urgency")) else r["policy_urgency"], axis=1)
        df["final_category"] = df.apply(
            lambda r: r["caseworker_category"] if pd.notna(r.get("caseworker_category")) else r["model_category"], axis=1)

        urgency_colours = {"Critical": "#B71C1C", "High": "#E65100", "Medium": "#F57F17", "Low": "#2E7D32"}

        # Row 1: Urgency and category
        st.subheader("Concern Distribution")
        tc1, tc2 = st.columns(2)

        with tc1:
            urg_counts = df["final_urgency"].value_counts().reindex(["Critical", "High", "Medium", "Low"]).fillna(0)
            fig_urg = go.Figure(data=[go.Bar(
                x=urg_counts.index,
                y=urg_counts.values,
                marker_color=[urgency_colours[u] for u in urg_counts.index],
            )])
            fig_urg.update_layout(title="Reports by urgency level", yaxis_title="Count",
                                  xaxis_title="", height=350, margin=dict(t=40, b=40, l=40, r=20),
                                  plot_bgcolor="white")
            st.plotly_chart(fig_urg, use_container_width=True)

        with tc2:
            cat_counts = df["final_category"].value_counts()
            fig_cat = px.bar(x=cat_counts.index, y=cat_counts.values,
                             color_discrete_sequence=["#004D40"])
            fig_cat.update_layout(title="Reports by category", yaxis_title="Count",
                                  xaxis_title="", height=350, margin=dict(t=40, b=40, l=40, r=20),
                                  plot_bgcolor="white", xaxis_tickangle=-45)
            st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("---")

        # Row 2: Timeline and units
        st.subheader("Reporting Patterns")
        tc3, tc4 = st.columns(2)

        with tc3:
            if pd.notna(df["submitted_at"]).any():
                timeline = df.copy()
                timeline["date"] = pd.to_datetime(timeline["submitted_at"]).dt.date
                daily_counts = timeline.groupby("date").size().reset_index(name="count")
                fig_time = px.line(daily_counts, x="date", y="count",
                                   color_discrete_sequence=["#004D40"])
                fig_time.update_layout(title="Reports over time", yaxis_title="Reports",
                                       xaxis_title="", height=350, margin=dict(t=40, b=40, l=40, r=20),
                                       plot_bgcolor="white")
                fig_time.update_traces(mode="lines+markers", line=dict(width=3), marker=dict(size=8))
                st.plotly_chart(fig_time, use_container_width=True)

        with tc4:
            unit_counts = df["unit_id"].value_counts().head(10)
            if not unit_counts.empty:
                fig_unit = px.bar(x=unit_counts.values, y=unit_counts.index, orientation="h",
                                  color_discrete_sequence=["#00695C"])
                fig_unit.update_layout(title="Top units by report volume", xaxis_title="Reports",
                                       yaxis_title="", height=350, margin=dict(t=40, b=40, l=40, r=20),
                                       plot_bgcolor="white", yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_unit, use_container_width=True)

        st.markdown("---")

        # Row 3: Demographics
        st.subheader("Demographics")
        tc5, tc6 = st.columns(2)

        with tc5:
            age_counts = df["age_band"].value_counts()
            age_colours = ["#004D40", "#00695C", "#00897B", "#26A69A", "#80CBC4"]
            fig_age = go.Figure(data=[go.Pie(
                labels=age_counts.index, values=age_counts.values,
                marker=dict(colors=age_colours[:len(age_counts)]),
                hole=0.4,
            )])
            fig_age.update_layout(title="Reports by age band", height=350,
                                  margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(fig_age, use_container_width=True)

        with tc6:
            role_counts = df["reporter_role"].value_counts()
            fig_role = go.Figure(data=[go.Pie(
                labels=role_counts.index, values=role_counts.values,
                marker=dict(colors=["#004D40", "#B71C1C", "#E65100", "#F57F17", "#2E7D32", "#1565C0"][:len(role_counts)]),
                hole=0.4,
            )])
            fig_role.update_layout(title="Reports by reporter role", height=350,
                                   margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(fig_role, use_container_width=True)

        st.markdown("---")

        # Key statistics
        st.subheader("Key Statistics")
        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.markdown(f"Category mismatch rate: **{df['category_mismatch'].mean():.0%}**")
        with stat2:
            reviewed_df = df[df["reviewed_at"].notna()]
            if len(reviewed_df) > 0:
                st.markdown(f"Urgency agreement (AI vs caseworker): **{(reviewed_df['caseworker_urgency'] == reviewed_df['model_urgency']).mean():.0%}**")
            else:
                st.markdown("Urgency agreement: **No reviews yet**")
        with stat3:
            if len(reviewed_df) > 0:
                st.markdown(f"Category agreement (AI vs caseworker): **{(reviewed_df['caseworker_category'] == reviewed_df['model_category']).mean():.0%}**")
            else:
                st.markdown("Category agreement: **No reviews yet**")

        name_links = get_all_name_links()
        if name_links:
            st.markdown("---")
            st.subheader("Linked Persons Summary")
            st.markdown(f"**{len(name_links)} individual(s)** appear in more than one report. See the Person Links page for details.")

    st.markdown("<div class='footer-text'>ACF Safeguarding Platform &mdash; AI-assisted decision support prototype &mdash; Research purposes only</div>", unsafe_allow_html=True)

elif page == "Person Links":
    st.title("Person Link Analysis")
    st.markdown("Individuals whose names appear across multiple safeguarding reports. Recurring names may indicate patterns of concern that warrant closer investigation.")

    name_links = get_all_name_links()

    if not name_links:
        st.info("No linked persons have been identified yet. Names are extracted automatically from concern narratives as reports are submitted.")
    else:
        st.markdown(f"**{len(name_links)} person(s)** appear in more than one report.")
        st.markdown("---")

        for link in name_links:
            with st.expander(f"\U0001F517 {link['name'].title()} \u2014 mentioned in {link['report_count']} reports (#{', #'.join(str(rid) for rid in sorted(link['report_ids']))})"):
                for rid in sorted(link["report_ids"]):
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("SELECT id, submitted_at, staff_category, policy_urgency, free_text, unit_id FROM reports WHERE id = %s", (rid,))
                    r = cur.fetchone()
                    cur.close()
                    conn.close()
                    if r:
                        ri = {"Critical": "\U0001F534", "High": "\U0001F7E0", "Medium": "\U0001F7E1", "Low": "\U0001F7E2"}.get(r[3], "")
                        st.markdown(f"{ri} **Report #{r[0]}** | {r[3]} | {r[2]} | Unit: {r[5]} | {r[1].strftime('%d %b %Y %H:%M') if r[1] else 'Unknown'}")
                        st.markdown(f"<div class='narrative-box' style='font-size: 0.9em;'>{r[4][:300]}{'...' if len(r[4]) > 300 else ''}</div>", unsafe_allow_html=True)
                st.markdown("---\n*Review these reports together to assess whether the pattern suggests escalating risk or a need for coordinated action.*")

    st.markdown("<div class='footer-text'>ACF Safeguarding Platform &mdash; AI-assisted decision support prototype &mdash; Research purposes only</div>", unsafe_allow_html=True)
