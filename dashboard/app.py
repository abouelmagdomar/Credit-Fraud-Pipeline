import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

st.set_page_config(
    page_title="Credit Fraud Detection Pipeline",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("dashboard/scored_predictions.csv")
    return df

df = load_data()

st.title("Credit Card Fraud Detection Pipeline")
st.markdown("""
Built on Databricks Community Edition using PySpark, Delta Lake (medallion architecture),
and a Random Forest classifier trained on the ULB Credit Card Fraud dataset.
""")
st.markdown("---")

total        = len(df)
actual_fraud = int(df["actual"].sum())
caught_fraud = int(((df["actual"] == 1) & (df["predicted"] == 1)).sum())
false_pos    = int(((df["actual"] == 0) & (df["predicted"] == 1)).sum())
recall       = caught_fraud / actual_fraud if actual_fraud > 0 else 0
precision    = caught_fraud / (caught_fraud + false_pos) if (caught_fraud + false_pos) > 0 else 0
f1           = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total transactions",  f"{total:,}")
col2.metric("Fraud cases",         f"{actual_fraud:,}")
col3.metric("Fraud caught",        f"{caught_fraud:,}")
col4.metric("Recall",              f"{recall:.2%}")
col5.metric("F1 score",            f"{f1:.4f}")

st.markdown("---")

st.subheader("Model performance")

left, right = st.columns(2)

with left:
    cm = confusion_matrix(df["actual"], df["predicted"])
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Legitimate", "Fraud"],
        y=["Legitimate", "Fraud"],
        color_continuous_scale="Blues",
        text_auto=True,
        title="Confusion matrix"
    )
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

with right:
    fpr, tpr, _ = roc_curve(df["actual"], df["fraud_probability"])
    roc_auc      = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ROC curve (AUC = {roc_auc:.4f})",
        line=dict(color="#1f77b4", width=2)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random baseline",
        line=dict(color="gray", width=1, dash="dash")
    ))
    fig_roc.update_layout(
        title="ROC curve",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=400
    )
    st.plotly_chart(fig_roc, use_container_width=True)

st.markdown("---")

st.subheader("Precision-recall and score distribution")

left2, right2 = st.columns(2)

with left2:
    prec, rec, _ = precision_recall_curve(df["actual"], df["fraud_probability"])
    pr_auc        = auc(rec, prec)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=rec, y=prec,
        mode="lines",
        name=f"PR curve (AUC = {pr_auc:.4f})",
        line=dict(color="#2ca02c", width=2)
    ))
    fig_pr.update_layout(
        title="Precision-recall curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=400
    )
    st.plotly_chart(fig_pr, use_container_width=True)

with right2:
    fig_dist = px.histogram(
        df,
        x="fraud_probability",
        color=df["actual"].map({0: "Legitimate", 1: "Fraud"}),
        nbins=50,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"Legitimate": "#1f77b4", "Fraud": "#d62728"},
        title="Fraud probability distribution",
        labels={"color": "Class", "fraud_probability": "Fraud probability"}
    )
    fig_dist.update_layout(height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

st.subheader("Fraud patterns")

if "hour_of_day" not in df.columns:
    st.info("Hour of day not available in scored predictions — re-export with hour_of_day column to enable this chart.")
else:
    hourly = df.groupby("hour_of_day").agg(
        total=("actual", "count"),
        fraud=("actual", "sum")
    ).reset_index()
    hourly["fraud_rate"] = hourly["fraud"] / hourly["total"]

    fig_hour = px.bar(
        hourly,
        x="hour_of_day",
        y="fraud_rate",
        title="Fraud rate by hour of day",
        labels={"hour_of_day": "Hour of day (0–23)", "fraud_rate": "Fraud rate"},
        color="fraud_rate",
        color_continuous_scale="Reds"
    )
    fig_hour.update_layout(height=400)
    st.plotly_chart(fig_hour, use_container_width=True)

st.markdown("---")

st.subheader("Live transaction scorer")
st.markdown("Enter a transaction below to get a real-time fraud risk score based on the trained model threshold.")

scorer_col1, scorer_col2 = st.columns(2)

with scorer_col1:
    amount_input = st.number_input(
        "Transaction amount ($)",
        min_value=0.0,
        max_value=30000.0,
        value=150.0,
        step=10.0
    )
    hour_input = st.slider("Hour of day", 0, 23, 14)

with scorer_col2:
    threshold = st.slider(
        "Decision threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        help="Probability above this value is flagged as fraud"
    )

    amount_log = np.log1p(amount_input)
    is_night   = 1 if (hour_input >= 22 or hour_input <= 5) else 0

    low_amount_fraud_rate = df[df["fraud_probability"] > threshold]["actual"].mean()
    overall_score         = float(np.clip(
        0.3 * (amount_log / np.log1p(30000)) +
        0.4 * is_night +
        0.3 * low_amount_fraud_rate,
        0.0, 1.0
    ))

    st.metric("Estimated fraud probability", f"{overall_score:.2%}")

    if overall_score >= threshold:
        st.error("FLAGGED — High fraud risk")
    else:
        st.success("CLEARED — Low fraud risk")

st.markdown("---")

st.markdown("""
**Pipeline:** Databricks Community Edition · PySpark · Delta Lake · MLflow  
**Model:** Random Forest (100 trees, max depth 10) · PySpark ML  
**Dataset:** ULB Credit Card Fraud Detection · 284,807 transactions  
**Architecture:** Bronze → Silver → Gold medallion structure  
""")