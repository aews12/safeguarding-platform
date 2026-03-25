import streamlit as st
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertModel, pipeline
import json
import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv(r"C:\Users\Alex\Documents\dissertation-ai\safeguarding_platform\.env")

MODEL_DIR = r"C:\Users\Alex\Documents\dissertation-ai\safeguarding_platform\models\multitask_v2"
DATABASE_URL = os.environ.get("DATABASE_URL")

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

# Full category list for caseworker corrections (matches model categories)
CASEWORKER_CATEGORIES = ["-- Accept AI suggestion --"] + CATEGORY_LABELS + ["Other"]
CASEWORKER_URGENCY = ["-- Accept AI suggestion --"] + URGENCY_LABELS


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
        urgency_logits = self.urgency_head(cls_output)
        category_logits = self.category_head(cls_output)
        return urgency_logits, category_logits


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
    state_dict = torch.load(
        os.path.join(MODEL_DIR, "model_state.pt"), map_location=device
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def load_ner_model():
    ner = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
    )
    return ner


# ---------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------

def predict(text):
    tokenizer, model, device = load_classification_model()

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        urg_logits, cat_logits = model(
            encoded["input_ids"], encoded["attention_mask"]
        )
        urg_probs = F.softmax(urg_logits, dim=-1).squeeze()
        cat_probs = F.softmax(cat_logits, dim=-1).squeeze()

    pred_urgency = id2label[torch.argmax(urg_probs).item()]
    urgency_conf = torch.max(urg_probs).item()

    # Primary category
    pred_category = id2category[torch.argmax(cat_probs).item()]
    category_conf = torch.max(cat_probs).item()

    # Secondary categories - anything above 15% confidence (excluding primary)
    secondary_categories = []
    for i, prob in enumerate(cat_probs.tolist()):
        cat_name = id2category[i]
        if cat_name != pred_category and prob >= 0.15:
            secondary_categories.append({
                "label": cat_name,
                "confidence": prob,
            })
    secondary_categories.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "urgency_label": pred_urgency,
        "urgency_confidence": urgency_conf,
        "category_label": pred_category,
        "category_confidence": category_conf,
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
                # Keep only the highest confidence instance of each name
                if name_lower not in seen_names or entity["score"] > seen_names[name_lower]["confidence"]:
                    seen_names[name_lower] = {
                        "name": name,
                        "confidence": float(entity["score"]),
                    }
    return list(seen_names.values())

# ---------------------------------------------------------
# Policy override logic
# ---------------------------------------------------------

def apply_policy_overrides(text, staff_category, base_urgency):
    t = text.lower().strip()

    order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    inv_order = {v: k for k, v in order.items()}

    current_level = order.get(base_urgency, 1)

    critical_keywords = [
        "kill myself", "end my life", "don't want to be alive",
        "suicide", "suicidal", "overdose", "hang myself",
        "jump off", "plan to hurt myself",
    ]

    physical_abuse_keywords = [
        "hit me", "hit them", "punched", "kicked", "beat me",
        "grabbed me by the neck", "threw me",
    ]

    sexual_abuse_keywords = [
        "touched me inappropriately", "touched them inappropriately",
        "private area", "warned me not to tell",
        "told me not to tell anyone", "raped",
    ]

    exploitation_keywords = [
        "much older individuals", "much older men",
        "gives me money", "pick me up late at night",
    ]

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

    high_min_categories = [
        "Abuse by adult in organisation",
        "Exploitation / trafficking",
        "FGM / harmful practices",
        "Sexual abuse / assault",
    ]
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
    cur.execute(
        """
        INSERT INTO reports (
            unit_id, reporter_role, location, age_band, channel,
            staff_category, free_text,
            model_urgency, model_urgency_confidence,
            model_category, model_category_confidence,
            policy_urgency, category_mismatch
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            data["unit_id"],
            data["reporter_role"],
            data["location"],
            data["age_band"],
            data["channel"],
            data["staff_category"],
            data["free_text"],
            data["model_urgency"],
            data["model_urgency_confidence"],
            data["model_category"],
            data["model_category_confidence"],
            data["policy_urgency"],
            data["category_mismatch"],
        ),
    )
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
        cur.execute(
            """
            INSERT INTO extracted_persons (report_id, person_name, confidence)
            VALUES (%s, %s, %s)
            """,
            (report_id, person["name"], person["confidence"]),
        )
    conn.commit()
    cur.close()
    conn.close()


def save_caseworker_review(report_id, urgency, category, notes, reviewer, status):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE reports
        SET caseworker_urgency = %s,
            caseworker_category = %s,
            caseworker_notes = %s,
            reviewed_by = %s,
            reviewed_at = NOW(),
            status = %s
        WHERE id = %s
        """,
        (urgency, category, notes, reviewer, status, report_id),
    )
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
        cur.execute(
            """
            SELECT ep.person_name, ep.report_id, r.submitted_at,
                   r.staff_category, r.policy_urgency, r.free_text
            FROM extracted_persons ep
            JOIN reports r ON ep.report_id = r.id
            WHERE LOWER(ep.person_name) = LOWER(%s)
            AND ep.report_id != %s
            ORDER BY r.submitted_at DESC
            """,
            (person["name"], report_id),
        )
        rows = cur.fetchall()
        for row in rows:
            matches.append({
                "matched_name": row[0],
                "report_id": row[1],
                "submitted_at": row[2],
                "category": row[3],
                "urgency": row[4],
                "narrative_preview": row[5][:150] + "..." if len(row[5]) > 150 else row[5],
            })
    cur.close()
    conn.close()
    return matches


def get_all_reports():
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT * FROM reports ORDER BY submitted_at DESC", conn
    )
    conn.close()
    return df


def get_persons_for_report(report_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT person_name, confidence FROM extracted_persons WHERE report_id = %s",
        (report_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"name": row[0], "confidence": row[1]} for row in rows]


def get_all_name_links():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT LOWER(person_name) as name, COUNT(DISTINCT report_id) as report_count,
               ARRAY_AGG(DISTINCT report_id) as report_ids
        FROM extracted_persons
        GROUP BY LOWER(person_name)
        HAVING COUNT(DISTINCT report_id) > 1
        ORDER BY report_count DESC
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"name": row[0], "report_count": row[1], "report_ids": row[2]} for row in rows]


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Safeguarding Platform",
    page_icon="\U0001F6E1",
    layout="wide",
)

# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------

st.sidebar.title("Safeguarding Platform")
page = st.sidebar.radio(
    "Select view",
    ["Submit a Concern", "Caseworker Dashboard", "Person Links"],
)


# ---------------------------------------------------------
# PAGE 1: Reporter submission form
# ---------------------------------------------------------

if page == "Submit a Concern":
    st.title("Report a Safeguarding Concern")
    st.markdown(
        "Use this form to record a welfare or safeguarding concern about "
        "a young person. Your report will be reviewed by the safeguarding team."
    )

    with st.form("concern_form"):
        st.subheader("Concern details")

        col1, col2, col3 = st.columns(3)

        with col1:
            unit_id = st.text_input("Unit ID", value="NW-ACF-001")
        with col2:
            reporter_role = st.selectbox(
                "Your role",
                [
                    "Adult Volunteer (CFAV)",
                    "Service Personnel",
                    "Parent / Carer",
                    "Family Member",
                    "Cadet",
                    "Other",
                ],
            )
        with col3:
            location = st.selectbox(
                "Where did this occur?",
                [
                    "Parade Night",
                    "Weekend Exercise",
                    "Annual Camp",
                    "Field Exercise",
                    "Online / Remote",
                    "Off-site Activity",
                    "Home",
                    "Other",
                ],
            )

        col4, col5 = st.columns(2)

        with col4:
            age_band = st.selectbox(
                "Age band of young person",
                ["Under 12", "12-14", "14-16", "16-18"],
            )
        with col5:
            channel = st.selectbox(
                "How was this reported to you?",
                [
                    "In person",
                    "Online form",
                    "Phone call",
                    "Email",
                    "Third party",
                    "Other",
                ],
            )

        staff_category = st.selectbox(
            "What type of concern is this?",
            REPORTER_CATEGORIES,
        )

        free_text = st.text_area(
            "Describe what has been observed or disclosed",
            height=200,
            placeholder="Include the full names of all individuals involved "
            "where known (e.g. the young person, any adults, other cadets). "
            "Describe what you saw or were told, including any specific "
            "language used. State the specific location where this occurred "
            "if applicable.",
        )

        submitted = st.form_submit_button("Submit concern")

    if submitted:
        if not free_text.strip():
            st.warning("Please describe the concern before submitting.")
        else:
            prediction = predict(free_text)

            policy_urgency = apply_policy_overrides(
                free_text, staff_category, prediction["urgency_label"]
            )

            staff_cat_lower = staff_category.lower().strip()
            model_cat_lower = prediction["category_label"].lower().strip()
            category_mismatch = (
                staff_cat_lower not in model_cat_lower
                and model_cat_lower not in staff_cat_lower
            )

            extracted_persons = extract_names(free_text)

            report_data = {
                "unit_id": unit_id,
                "reporter_role": reporter_role,
                "location": location,
                "age_band": age_band,
                "channel": channel,
                "staff_category": staff_category,
                "free_text": free_text,
                "model_urgency": prediction["urgency_label"],
                "model_urgency_confidence": prediction["urgency_confidence"],
                "model_category": prediction["category_label"],
                "model_category_confidence": prediction["category_confidence"],
                "policy_urgency": policy_urgency,
                "category_mismatch": category_mismatch,
            }

            try:
                report_id = save_report(report_data)
                save_extracted_persons(report_id, extracted_persons)

                st.success(
                    f"Your concern has been received and will be reviewed "
                    f"by the safeguarding team.\n\n"
                    f"**Reference number: {report_id}**\n\n"
                    f"Please keep this reference number for your records. "
                    f"If the situation changes or you have further information, "
                    f"please submit a new report and quote this reference."
                )
            except Exception as e:
                st.error(
                    "There was a problem saving your report. "
                    "Please try again or contact the safeguarding team directly."
                )
                st.exception(e)


# ---------------------------------------------------------
# PAGE 2: Caseworker dashboard
# ---------------------------------------------------------

elif page == "Caseworker Dashboard":
    st.title("Caseworker Dashboard")
    st.markdown(
        "Overview of submitted safeguarding concerns with AI-assisted triage."
    )

    df = get_all_reports()

    if df.empty:
        st.info("No reports have been submitted yet.")
    else:
        # Summary metrics
        st.subheader("Summary")
        met_col1, met_col2, met_col3, met_col4, met_col5, met_col6 = st.columns(6)

        with met_col1:
            st.metric("Total Reports", len(df))
        with met_col2:
            st.metric("Critical", len(df[df["policy_urgency"] == "Critical"]))
        with met_col3:
            st.metric("High", len(df[df["policy_urgency"] == "High"]))
        with met_col4:
            st.metric("Medium", len(df[df["policy_urgency"] == "Medium"]))
        with met_col5:
            st.metric("Low", len(df[df["policy_urgency"] == "Low"]))
        with met_col6:
            reviewed = len(df[df["reviewed_at"].notna()])
            st.metric("Reviewed", f"{reviewed}/{len(df)}")

        st.markdown("---")

        # Filters
        st.subheader("Filters")
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            urgency_filter = st.multiselect(
                "Urgency",
                options=["Critical", "High", "Medium", "Low"],
                default=["Critical", "High", "Medium", "Low"],
            )
        with filter_col2:
            mismatch_filter = st.selectbox(
                "Category mismatch",
                options=["All", "Mismatches only", "No mismatches"],
            )
        with filter_col3:
            status_filter = st.selectbox(
                "Status",
                options=["All", "Open", "In progress", "Closed"],
            )
        with filter_col4:
            review_filter = st.selectbox(
                "Review status",
                options=["All", "Awaiting review", "Reviewed"],
            )

        # Apply filters
        filtered = df[df["policy_urgency"].isin(urgency_filter)]

        if mismatch_filter == "Mismatches only":
            filtered = filtered[filtered["category_mismatch"] == True]
        elif mismatch_filter == "No mismatches":
            filtered = filtered[filtered["category_mismatch"] == False]

        if status_filter != "All":
            filtered = filtered[filtered["status"] == status_filter]

        if review_filter == "Awaiting review":
            filtered = filtered[filtered["reviewed_at"].isna()]
        elif review_filter == "Reviewed":
            filtered = filtered[filtered["reviewed_at"].notna()]

        st.markdown(f"Showing **{len(filtered)}** of **{len(df)}** reports")
        st.markdown("---")

        # Display each report
        for _, row in filtered.iterrows():
            urgency_colours = {
                "Critical": "\U0001F534",
                "High": "\U0001F7E0",
                "Medium": "\U0001F7E1",
                "Low": "\U0001F7E2",
            }
            urgency_icon = urgency_colours.get(row["policy_urgency"], "")

            reviewed_tag = " \u2705" if pd.notna(row.get("reviewed_at")) else ""

            with st.expander(
                f"{urgency_icon} Report #{row['id']} | "
                f"{row['policy_urgency']} urgency | "
                f"{row['staff_category']} | "
                f"{row['submitted_at'].strftime('%d %b %Y %H:%M') if pd.notna(row['submitted_at']) else 'Unknown date'}"
                f"{reviewed_tag}"
            ):
                detail_left, detail_right = st.columns([2, 1])

                with detail_left:
                    st.markdown("**Concern narrative:**")
                    st.markdown(
                        f"<div style='background-color: #f0f2f6; padding: 15px; "
                        f"border-radius: 5px; margin-bottom: 10px;'>"
                        f"{row['free_text']}</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"**Reporter:** {row['reporter_role']} | "
                        f"**Unit:** {row['unit_id']} | "
                        f"**Location:** {row['location']} | "
                        f"**Age band:** {row['age_band']} | "
                        f"**Channel:** {row['channel']}"
                    )

                with detail_right:
                    st.markdown("**AI Assessment:**")

                    st.markdown(
                        f"Urgency: **{row['model_urgency']}** "
                        f"({row['model_urgency_confidence']:.0%} confident)"
                    )

                    if row["policy_urgency"] != row["model_urgency"]:
                        st.markdown(
                            f"Policy-adjusted: **{row['policy_urgency']}** "
                            f"(elevated by safety rules)"
                        )

                    st.markdown("---")

                    st.markdown(
                        f"AI category: **{row['model_category']}** "
                        f"({row['model_category_confidence']:.0%} confident)"
                    )

                    # Show secondary categories if available
                    # Re-run prediction to get secondary categories
                    secondary = predict(row["free_text"]).get("secondary_categories", [])
                    if secondary:
                        sec_text = ", ".join(
                            f"{s['label']} ({s['confidence']:.0%})"
                            for s in secondary
                        )
                        st.markdown(f"Also flagged: {sec_text}")

                    st.markdown(
                        f"Reporter category: **{row['staff_category']}**"
                    )

                    if row["category_mismatch"]:
                        st.warning(
                            "Category mismatch detected. The AI suggests a "
                            "different category to the reporter. Please review."
                        )

                    st.markdown("---")

                    # Show extracted persons
                    persons = get_persons_for_report(row["id"])
                    if persons:
                        st.markdown("**Persons identified in narrative:**")
                        for p in persons:
                            other_mentions = find_name_matches(row["id"], [p])
                            if other_mentions:
                                st.error(
                                    f"\U0001F517 **{p['name']}** "
                                    f"({p['confidence']:.0%}) \u2014 also mentioned "
                                    f"in {len(other_mentions)} other report(s)"
                                )
                            else:
                                st.markdown(
                                    f"\U0001F464 {p['name']} "
                                    f"({p['confidence']:.0%})"
                                )

                # Caseworker review section
                st.markdown("---")

                if pd.notna(row.get("reviewed_at")):
                    st.markdown("**Caseworker Review (completed):**")
                    rev_col1, rev_col2 = st.columns(2)
                    with rev_col1:
                        st.markdown(f"Caseworker urgency: **{row['caseworker_urgency']}**")
                        st.markdown(f"Caseworker category: **{row['caseworker_category']}**")
                    with rev_col2:
                        st.markdown(f"Reviewed by: **{row['reviewed_by']}**")
                        st.markdown(
                            f"Reviewed at: **{row['reviewed_at'].strftime('%d %b %Y %H:%M') if pd.notna(row['reviewed_at']) else ''}**"
                        )
                    if row.get("caseworker_notes"):
                        st.markdown(f"Notes: {row['caseworker_notes']}")

                    # Show agreement/disagreement with model
                    urg_agree = row["caseworker_urgency"] == row["model_urgency"]
                    cat_agree = row["caseworker_category"] == row["model_category"]
                    if urg_agree and cat_agree:
                        st.success("Caseworker agreed with both AI urgency and category.")
                    elif not urg_agree and not cat_agree:
                        st.info(
                            f"Caseworker adjusted urgency from {row['model_urgency']} "
                            f"to {row['caseworker_urgency']} and category from "
                            f"{row['model_category']} to {row['caseworker_category']}."
                        )
                    elif not urg_agree:
                        st.info(
                            f"Caseworker adjusted urgency from {row['model_urgency']} "
                            f"to {row['caseworker_urgency']}."
                        )
                    else:
                        st.info(
                            f"Caseworker adjusted category from {row['model_category']} "
                            f"to {row['caseworker_category']}."
                        )
                else:
                    st.markdown("**Caseworker Review:**")

                    # Use unique keys based on report ID for each form element
                    rid = row["id"]

                    rev_col1, rev_col2 = st.columns(2)

                    with rev_col1:
                        cw_urgency = st.selectbox(
                            "Your urgency assessment",
                            options=CASEWORKER_URGENCY,
                            key=f"cw_urg_{rid}",
                        )
                        cw_category = st.selectbox(
                            "Your category assessment",
                            options=CASEWORKER_CATEGORIES,
                            key=f"cw_cat_{rid}",
                        )

                    with rev_col2:
                        cw_reviewer = st.text_input(
                            "Your name",
                            key=f"cw_name_{rid}",
                        )
                        cw_status = st.selectbox(
                            "Update status",
                            options=["Open", "In progress", "Closed"],
                            key=f"cw_status_{rid}",
                        )

                    cw_notes = st.text_area(
                        "Caseworker notes (optional)",
                        key=f"cw_notes_{rid}",
                        height=80,
                        placeholder="Record your assessment, actions taken, or rationale for any changes.",
                    )

                    if st.button("Save review", key=f"cw_save_{rid}"):
                        if not cw_reviewer.strip():
                            st.warning("Please enter your name before saving the review.")
                        else:
                            # Resolve "accept AI suggestion" options
                            final_urgency = row["model_urgency"] if cw_urgency == "-- Accept AI suggestion --" else cw_urgency
                            final_category = row["model_category"] if cw_category == "-- Accept AI suggestion --" else cw_category

                            save_caseworker_review(
                                report_id=rid,
                                urgency=final_urgency,
                                category=final_category,
                                notes=cw_notes,
                                reviewer=cw_reviewer,
                                status=cw_status,
                            )
                            st.success(f"Review saved for Report #{rid}.")
                            st.rerun()

        st.markdown("---")

        # Trend analysis
        st.subheader("Trend Analysis")

        if len(df) >= 2:
            trend_col1, trend_col2 = st.columns(2)

            with trend_col1:
                st.markdown("**Reports by urgency level**")
                urgency_counts = df["policy_urgency"].value_counts()
                urgency_order = ["Critical", "High", "Medium", "Low"]
                urgency_counts = urgency_counts.reindex(urgency_order).fillna(0)
                st.bar_chart(urgency_counts)

            with trend_col2:
                st.markdown("**Reports by AI-predicted category**")
                cat_counts = df["model_category"].value_counts()
                st.bar_chart(cat_counts)

            if len(df) > 0:
                mismatch_rate = df["category_mismatch"].mean()
                st.markdown(
                    f"**Category mismatch rate:** {mismatch_rate:.0%} of reports "
                    f"have a mismatch between the reporter's category and the "
                    f"AI-predicted category."
                )

            # Model agreement stats (only for reviewed reports)
            reviewed_df = df[df["reviewed_at"].notna()]
            if len(reviewed_df) > 0:
                st.markdown("---")
                st.markdown("**Model vs caseworker agreement (reviewed reports only):**")
                agree_col1, agree_col2 = st.columns(2)
                with agree_col1:
                    urg_agreement = (reviewed_df["caseworker_urgency"] == reviewed_df["model_urgency"]).mean()
                    st.markdown(f"Urgency agreement: **{urg_agreement:.0%}**")
                with agree_col2:
                    cat_agreement = (reviewed_df["caseworker_category"] == reviewed_df["model_category"]).mean()
                    st.markdown(f"Category agreement: **{cat_agreement:.0%}**")
        else:
            st.info("More reports are needed to show trend analysis.")


# ---------------------------------------------------------
# PAGE 3: Person links
# ---------------------------------------------------------

elif page == "Person Links":
    st.title("Person Link Analysis")
    st.markdown(
        "This view identifies individuals whose names appear across multiple "
        "safeguarding reports. Recurring names may indicate patterns of concern "
        "that warrant closer investigation."
    )

    name_links = get_all_name_links()

    if not name_links:
        st.info(
            "No linked persons have been identified yet. Names are extracted "
            "automatically from concern narratives as reports are submitted."
        )
    else:
        st.markdown(f"**{len(name_links)} person(s)** appear in more than one report.")
        st.markdown("---")

        for link in name_links:
            with st.expander(
                f"\U0001F517 {link['name'].title()} \u2014 mentioned in "
                f"{link['report_count']} reports "
                f"(#{', #'.join(str(rid) for rid in sorted(link['report_ids']))})"
            ):
                for rid in sorted(link["report_ids"]):
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute(
                        """
                        SELECT id, submitted_at, staff_category, policy_urgency,
                               free_text, unit_id
                        FROM reports WHERE id = %s
                        """,
                        (rid,),
                    )
                    row = cur.fetchone()
                    cur.close()
                    conn.close()

                    if row:
                        urgency_colours = {
                            "Critical": "\U0001F534",
                            "High": "\U0001F7E0",
                            "Medium": "\U0001F7E1",
                            "Low": "\U0001F7E2",
                        }
                        icon = urgency_colours.get(row[3], "")

                        st.markdown(
                            f"{icon} **Report #{row[0]}** | "
                            f"{row[3]} urgency | {row[2]} | "
                            f"Unit: {row[5]} | "
                            f"{row[1].strftime('%d %b %Y %H:%M') if row[1] else 'Unknown date'}"
                        )
                        st.markdown(
                            f"<div style='background-color: #f0f2f6; padding: 10px; "
                            f"border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;'>"
                            f"{row[4][:300]}{'...' if len(row[4]) > 300 else ''}</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    "---\n"
                    "*Review these reports together to assess whether the "
                    "pattern suggests escalating risk or a need for coordinated action.*"
                )