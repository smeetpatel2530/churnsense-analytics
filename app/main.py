import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="ChurnSense Analytics", page_icon="📊", layout="wide")

@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "monthly_charges": np.random.uniform(20, 120, n).round(2),
        "total_charges": np.random.uniform(100, 8000, n).round(2),
        "contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
        "internet_service": np.random.choice(["Fiber optic", "DSL", "No"], n, p=[0.5, 0.35, 0.15]),
        "support_tickets": np.random.randint(0, 8, n),
        "churn": np.random.choice(["Yes", "No"], n, p=[0.27, 0.73])
    })
    return df


def simple_risk_score(tenure, monthly_charges, support_tickets, contract):
    score = 0
    if tenure < 12:
        score += 30
    if monthly_charges > 80:
        score += 25
    if support_tickets >= 4:
        score += 25
    if contract == "Month-to-month":
        score += 20
    return min(score, 100)


df = load_sample_data()

st.title("📊 ChurnSense Analytics — Customer Churn Prediction Dashboard")
st.markdown("Interactive predictive analytics dashboard for churn analysis, business insights, and customer-level risk scoring.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", len(df))
col2.metric("Churn Rate", f"{(df['churn'].eq('Yes').mean()*100):.1f}%")
col3.metric("Avg Monthly Charges", f"${df['monthly_charges'].mean():.2f}")
col4.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")

st.subheader("Churn Insights")
left, right = st.columns(2)
with left:
    fig1 = px.histogram(df, x="contract", color="churn", barmode="group", title="Churn by Contract Type")
    st.plotly_chart(fig1, use_container_width=True)
with right:
    fig2 = px.scatter(df, x="tenure", y="monthly_charges", color="churn", title="Tenure vs Monthly Charges")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Customer Risk Predictor")
with st.form("risk_form"):
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges = st.slider("Monthly Charges", 20, 150, 85)
    with c2:
        support_tickets = st.slider("Support Tickets", 0, 10, 3)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    risk = simple_risk_score(tenure, monthly_charges, support_tickets, contract)
    if risk >= 70:
        label = "High Risk"
    elif risk >= 40:
        label = "Medium Risk"
    else:
        label = "Low Risk"
    st.success(f"Predicted churn risk: {risk}% — {label}")

st.markdown("---")
st.caption("Built by Smeet Patel | Streamlit + Plotly + Scikit-learn")
