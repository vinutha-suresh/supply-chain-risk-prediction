import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Supply Chain Risk Prediction",
    page_icon="📦",
    layout="wide"
)

# ==================================================
# LOAD FILES
# ==================================================

df = pd.read_csv("supply_chain.csv")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
country_encoder = joblib.load("country_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

comparison_df = pd.read_csv("model_comparison.csv")

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1,h2,h3 {
    color: #003366;
}

[data-testid="stMetric"] {
    background-color: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 1px 1px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("📦 Supply Chain Risk")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset Analysis",
        "Model Comparison",
        "Prediction"
    ]
)

# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.title("📦 Supply Chain Risk Prediction Dashboard")

    total_records = len(df)

    total_features = len(df.columns) - 1

    total_countries = df["supplier_country"].nunique()

    best_model = comparison_df.iloc[0]["Model"]

    best_accuracy = round(
        comparison_df.iloc[0]["Accuracy"] * 100,
        2
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        total_records
    )

    col2.metric(
        "Features",
        total_features
    )

    col3.metric(
        "Countries",
        total_countries
    )

    col4.metric(
        "Best Model",
        best_model
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    risk_counts = df["risk_classification"].value_counts()

    fig1 = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Risk Distribution"
    )

    col1.plotly_chart(
        fig1,
        use_container_width=True
    )

    country_count = (
        df["supplier_country"]
        .value_counts()
        .head(10)
    )

    fig2 = px.bar(
        x=country_count.index,
        y=country_count.values,
        title="Top 10 Supplier Countries"
    )

    col2.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Best Model Performance")

    st.success(
        f"""
        Best Model : {best_model}

        Accuracy : {best_accuracy}%
        """
    )

# ==================================================
# DATASET ANALYSIS
# ==================================================

elif page == "Dataset Analysis":

    st.title("📊 Dataset Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.markdown("---")

    st.subheader("Dataset Statistics")

    st.dataframe(df.describe())

    st.markdown("---")

    st.subheader("Risk Class Distribution")

    fig = px.histogram(
        df,
        x="risk_classification",
        color="risk_classification"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Shipping Cost Distribution")

    fig = px.histogram(
        df,
        x="shipping_costs"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# MODEL COMPARISON
# ==================================================

elif page == "Model Comparison":

    st.title("🤖 Model Comparison")

    st.subheader("Performance Table")

    st.dataframe(comparison_df)

    st.markdown("---")

    fig1 = px.bar(
        comparison_df,
        x="Model",
        y="Accuracy",
        title="Accuracy Comparison",
        text="Accuracy"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.bar(
        comparison_df,
        x="Model",
        y="Precision",
        title="Precision Comparison",
        text="Precision"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    fig3 = px.bar(
        comparison_df,
        x="Model",
        y="Recall",
        title="Recall Comparison",
        text="Recall"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    fig4 = px.bar(
        comparison_df,
        x="Model",
        y="F1 Score",
        title="F1 Score Comparison",
        text="F1 Score"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# ==================================================
# PREDICTION
# ==================================================

elif page == "Prediction":

    st.title("🔮 Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:

        warehouse_inventory_level = st.number_input(
            "Warehouse Inventory Level",
            value=500.0
        )

        handling_equipment_availability = st.number_input(
            "Handling Equipment Availability",
            value=0.5
        )

        order_fulfillment_status = st.number_input(
            "Order Fulfillment Status",
            value=0.5
        )

        weather_condition_severity = st.number_input(
            "Weather Condition Severity",
            value=0.5
        )

        shipping_costs = st.number_input(
            "Shipping Costs",
            value=200.0
        )

    with col2:

        supplier_reliability_score = st.number_input(
            "Supplier Reliability Score",
            value=0.5
        )

        historical_demand = st.number_input(
            "Historical Demand",
            value=1000.0
        )

        cargo_condition_status = st.number_input(
            "Cargo Condition Status",
            value=0.5
        )

        country = st.selectbox(
            "Supplier Country",
            list(country_encoder.classes_)
        )

    if st.button("Predict Risk"):

        country_encoded = (
            country_encoder.transform([country])[0]
        )

        input_data = np.array([[
            warehouse_inventory_level,
            handling_equipment_availability,
            order_fulfillment_status,
            weather_condition_severity,
            shipping_costs,
            supplier_reliability_score,
            historical_demand,
            cargo_condition_status,
            country_encoded
        ]])

        input_scaled = scaler.transform(
            input_data
        )

        prediction = model.predict(
            input_scaled
        )

        result = (
            target_encoder.inverse_transform(
                prediction
            )[0]
        )

        st.markdown("---")

        if result == "High Risk":

            st.error(
                f"Predicted Risk Level : {result}"
            )

        elif result == "Moderate Risk":

            st.warning(
                f"Predicted Risk Level : {result}"
            )

        else:

            st.success(
                f"Predicted Risk Level : {result}"
            )