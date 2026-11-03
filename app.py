
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Kenya Food Insecurity Early-Warning System",
    page_icon="🌾",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

package = joblib.load("kenya_food_insecurity_geo_model.pkl")

model = package["model"]
features = package["features"]
threshold = package["threshold"]
county_to_admin2 = package["county_to_admin2"]

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌾 Kenya Food Insecurity Early-Warning System")

st.markdown(
    """
    This machine-learning prototype estimates whether conditions in a
    Kenyan area are associated with **IPC Phase 3 or higher food-insecurity risk**.

    The model uses geographic, environmental, demographic, economic,
    and conflict-related indicators.
    """
)

st.divider()

# --------------------------------------------------
# LOCATION
# --------------------------------------------------

st.subheader("📍 Location")

col1, col2 = st.columns(2)

with col1:
    county = st.selectbox(
        "County",
        options=sorted(county_to_admin2.keys())
    )

with col2:
    admin2 = st.selectbox(
        "Admin-2 Area",
        options=county_to_admin2[county]
    )

# --------------------------------------------------
# INDICATORS
# --------------------------------------------------

st.subheader("📊 Area Indicators")

col1, col2 = st.columns(2)

with col1:

    population = st.number_input(
        "Population",
        min_value=0,
        value=150000,
        step=1000,
        help="Estimated population of the selected administrative area."
    )

    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda x: [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ][x - 1],
        help=(
            "Month of assessment. Month provides seasonal context to the model. "
            "The model was trained on observations recorded in February, June, "
            "and October. Predictions for other months should therefore be "
            "interpreted with additional caution."
        )
    )
    conflict = st.number_input(
        "Conflict Fatalities",
        min_value=0.0,
        value=0.30,
        format="%.4f",
        help=(
            "Conflict-related fatalities associated with neighbouring "
            "Admin-2 areas. Higher values represent greater conflict intensity."
        )
    )

    drought_ndvi = st.number_input(
        "Vegetation Condition (NDVI)",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        format="%.4f",
        help=(
            "Normalized Difference Vegetation Index (NDVI), which measures "
            "vegetation greenness and condition. Lower values indicate poorer "
            "vegetation conditions and may reflect greater drought stress."
        )
    )


with col2:

    drought_rainfall = st.number_input(
        "Average Rainfall per Dekad (mm)",
        min_value=0.0,
        value=30.0,
        format="%.4f",
        help=(
            "Average rainfall received per dekad (approximately a 10-day period), "
            "measured in millimetres. Lower rainfall can indicate drier conditions."
        )
    )

    exchange_rate = st.number_input(
        "Exchange Rate (Local Currency per USD)",
        min_value=0.0,
        value=110.0,
        format="%.4f",
        help=(
            "Exchange rate expressed as units of local currency per US dollar. "
            "Exchange-rate movements can influence the cost of imported food, "
            "fuel and agricultural inputs."
        )
    )

    food_prices = st.number_input(
        "Food Price Index",
        min_value=0.0,
        value=1.20,
        format="%.4f",
        help=(
            "Index representing the price level of food commodities. "
            "Higher values generally indicate higher food-price conditions."
        )
    )

    trade_food = st.number_input(
        "Food Trade Indicator",
        value=0.50,
        format="%.4f",
        help=(
            "Indicator representing conditions in food-commodity trade and "
            "market flows. It summarizes changes in food-trade conditions "
            "rather than representing a monetary price or quantity directly."
        )
    )

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "Assess IPC3+ Risk",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "adm1_name": [county],
        "adm2_name": [admin2],
        "population": [population],
        "month": [month],
        "conflict_original": [conflict],
        "drought_ndvi_original": [drought_ndvi],
        "drought_rainfall_original": [drought_rainfall],
        "exchange_rates_original": [exchange_rate],
        "food_prices_original": [food_prices],
        "trade_food_original": [trade_food]
    })

    input_data = input_data[features]

    score = model.predict_proba(input_data)[:, 1][0]

    prediction = int(score >= threshold)

    st.subheader("Assessment Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ Elevated IPC3+ Risk"
            )

            st.write(
                f"""
                The model identifies the entered conditions for
                **{admin2}, {county} County** as associated with
                elevated risk of IPC Phase 3 or higher food insecurity.
                """
            )

        else:

            st.success(
                "✅ Lower IPC3+ Risk"
            )

            st.write(
                f"""
                The model identifies the entered conditions for
                **{admin2}, {county} County** as associated with
                lower risk of IPC Phase 3 or higher food insecurity.
                """
            )

    with result_col2:

        st.metric(
            "Model Risk Score",
            f"{score:.4f}"
        )

        st.metric(
            "Decision Threshold",
            f"{threshold:.3f}"
        )

    with st.expander("How does the model evaluate IPC3+ risk?"):

      st.markdown(
        """
        The model combines **geographic, demographic, environmental,
        conflict, and economic indicators** to estimate whether the
        selected area shows conditions associated with **IPC Phase 3
        or higher food insecurity**.

        ### Predictors used by the model

        **📍 County**  
        The county (Admin-1) in which the area is located. Geography helps
        the model account for differences in historical food-security
        patterns and local conditions across Kenya.

        **📍 Admin-2 Area**  
        The smaller administrative area within the selected county.
        This provides more localized geographic information.

        **👥 Population**  
        The estimated population of the selected area.

        **📅 Month**  
        The month being assessed. This allows the model to account for
        seasonal differences in food-security conditions.

        **⚔️ Conflict Indicator**  
        Represents conflict-related conditions recorded for the area.
        Conflict can disrupt livelihoods, markets, agricultural activity,
        movement, and access to food.

        **🌱 Vegetation Condition (NDVI)**  
        NDVI means Normalized Difference Vegetation Index and represents
        vegetation condition or greenness. Lower values can indicate poorer
        vegetation conditions and possible drought stress.

        **🌧️ Drought Rainfall Indicator**  
        Represents rainfall conditions associated with drought monitoring.
        Lower rainfall conditions may contribute to reduced crop and
        livestock productivity.

        **💱 Exchange Rate Indicator**  
        Represents exchange-rate conditions. Exchange-rate changes can
        influence the cost of imported food, agricultural inputs, fuel,
        and other goods.

        **🍚 Food Price Indicator**  
        Represents food-price conditions. Higher food prices can reduce
        households' ability to access adequate food.

        **🚚 Food Trade Indicator**  
        Represents conditions related to food trade and market flows.
        Disruptions can affect local food availability and access.

        ### How the prediction is made

        The Gradient Boosting model evaluates all these predictors
        **together** rather than using any single indicator to determine risk.

        The model produces a risk score and compares it with the validated
        decision threshold of **0.035**.

        - **Score ≥ 0.035 → Elevated IPC3+ Risk**
        - **Score < 0.035 → Lower IPC3+ Risk**

        The threshold was selected using validation data while maintaining
        at least **80% recall** for IPC3+ observations.

        **Important:** This model identifies patterns associated with IPC3+
        conditions. It does not establish that an individual predictor
        causes food insecurity, and the prediction is not an official IPC
        classification.
        """
    )

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

with st.expander("Model performance"):

    st.markdown(
        """
        **Final chronological test period: 2024–2026**

        - Accuracy: **87.7%**
        - Precision for IPC3+: **36.9%**
        - Recall for IPC3+: **86.5%**
        - F1-score: **0.518**
        - ROC-AUC: **0.933**
        - PR-AUC: **0.469**

        The model correctly detected **96 of 111 IPC3+ observations**
        in the final test period.
        """
    )

st.caption(
    "Kenya Food Insecurity Early-Warning System — Machine Learning Prototype"
)
