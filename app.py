from pathlib import Path
import html
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
AMENITIES = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]

st.set_page_config(
    page_title="EstateScope | Property Advisor",
    page_icon="🏡",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1220;
        color: #e8eef7;
    }

    [data-testid="stSidebar"] {
        background: #101b2e;
        border-right: 1px solid #24334b;
    }

    [data-testid="stSidebar"] * {
        color: #e8eef7;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #f1f5fb !important;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: #c3cfe0;
    }

    .hero {
        background: linear-gradient(120deg, #142b45, #123c43);
        border: 1px solid #28515d;
        border-radius: 22px;
        padding: 30px;
        margin-bottom: 24px;
    }

    .hero .eyebrow {
        color: #65dbc3;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .hero h1 {
        margin: 0 0 10px 0;
        font-size: 38px;
    }

    .hero p {
        margin: 0;
        color: #d0e2e8;
        max-width: 850px;
    }

    [data-testid="stMetric"] {
        background: #142137;
        border: 1px solid #283b55;
        padding: 18px;
        border-radius: 16px;
    }

    [data-testid="stMetricValue"] {
        color: #f3ce85;
    }

    [data-testid="stMetricLabel"] {
        color: #c3cfe0;
    }

    [data-testid="stForm"] {
        background: #101c2f;
        border: 1px solid #2a3b54;
        border-radius: 18px;
        padding: 20px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #3f8e86;
        background: #174d4c;
        color: white;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #76e2c5;
        color: #76e2c5;
    }

    .status-good {
        background: #103d35;
        border: 1px solid #2f8a70;
        border-radius: 14px;
        padding: 18px;
        margin: 12px 0;
        color: #a1f1cc;
    }

    .status-review {
        background: #40331e;
        border: 1px solid #8b6d36;
        border-radius: 14px;
        padding: 18px;
        margin: 12px 0;
        color: #f4d99c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

plt.rcParams.update({
    "figure.facecolor": "#101c2f",
    "axes.facecolor": "#101c2f",
    "axes.edgecolor": "#42546f",
    "axes.labelcolor": "#cbd5e1",
    "text.color": "#e8eef7",
    "xtick.color": "#b9c8db",
    "ytick.color": "#b9c8db",
    "grid.color": "#2a3b54",
    "font.size": 10
})


# =========================================================
# Loading
# =========================================================

@st.cache_resource
def load_models():
    return (
        joblib.load(
            BASE_DIR / "models" / "classification_pipeline.joblib"
        ),
        joblib.load(
            BASE_DIR / "models" / "regression_pipeline.joblib"
        )
    )


@st.cache_data
def load_data():
    data = pd.read_csv(
        BASE_DIR / "data" / "india_housing_prices_cleaned.csv"
    )

    benchmarks = pd.read_csv(
        BASE_DIR / "data" / "price_benchmarks.csv"
    )

    with open(
        BASE_DIR / "models" / "metadata.json",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    with open(
        BASE_DIR / "reports" / "test_metrics.json",
        encoding="utf-8"
    ) as file:
        metrics = json.load(file)

    return data, benchmarks, metadata, metrics


try:
    classifier, regressor = load_models()
    properties, benchmarks, metadata, metrics = load_data()
except Exception as error:
    st.error(
        "Could not load the project files. Confirm that data, models "
        "and reports are beside app.py and that package versions match."
    )
    st.exception(error)
    st.stop()

GROWTH = float(metadata["assumed_annual_growth"])
YEARS = int(metadata["projection_years"])
REFERENCE_YEAR = int(metadata["dataset_reference_year"])


# =========================================================
# Helpers
# =========================================================

def hero(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_chart(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def benchmark_for(state, city, property_type):
    match = benchmarks.loc[
        benchmarks["State"].eq(state)
        & benchmarks["City"].eq(city)
        & benchmarks["Property_Type"].eq(property_type),
        "Benchmark_Price_per_SqFt"
    ]

    if match.empty:
        return float(metadata["fallback_benchmark"]), True

    return float(match.iloc[0]), False


def analyze(values):
    input_df = pd.DataFrame([values])[metadata["feature_columns"]]

    prediction = int(classifier.predict(input_df)[0])
    projected_price = float(regressor.predict(input_df)[0])

    positive_index = list(classifier.classes_).index(1)

    probability = float(
        classifier.predict_proba(input_df)[0, positive_index]
    )

    price_per_sqft = round(
        values["Price_in_Lakhs"] * 100000 / values["Size_in_SqFt"],
        2
    )

    benchmark, fallback = benchmark_for(
        values["State"],
        values["City"],
        values["Property_Type"]
    )

    return {
        "inputs": values.copy(),
        "prediction": prediction,
        "projected_price": projected_price,
        "probability": probability,
        "price_per_sqft": price_per_sqft,
        "benchmark": benchmark,
        "fallback": fallback,
        "rule_result": bool(price_per_sqft <= benchmark),
        "formula_price": values["Price_in_Lakhs"] * (1 + GROWTH) ** YEARS
    }


def select_category(label, column, default):
    options = sorted(properties[column].dropna().unique().tolist())
    index = options.index(default) if default in options else 0

    return st.selectbox(
        label,
        options,
        index=index
    )


def analysis_report(result):
    values = result["inputs"]

    selected_amenities = [
        amenity for amenity in AMENITIES
        if values[f"Has_{amenity}"] == 1
    ]

    lines = [
        "ESTATESCOPE PROPERTY ANALYSIS",
        "",
        "SUBMITTED PROPERTY",
        f"State: {values['State']}",
        f"City: {values['City']}",
        f"Property type: {values['Property_Type']}",
        f"BHK: {values['BHK']}",
        f"Size: {values['Size_in_SqFt']} sq ft",
        f"Current price: INR {values['Price_in_Lakhs']:,.2f} lakh",
        f"Age at {REFERENCE_YEAR}: {values['Age_of_Property']} years",
        f"Furnishing: {values['Furnished_Status']}",
        f"Transport access: {values['Public_Transport_Accessibility']}",
        f"Nearby schools: {values['Nearby_Schools']}",
        f"Nearby hospitals: {values['Nearby_Hospitals']}",
        f"Parking: {values['Parking_Space']}",
        f"Security: {values['Security']}",
        f"Facing: {values['Facing']}",
        f"Owner type: {values['Owner_Type']}",
        f"Availability: {values['Availability_Status']}",
        f"Amenities: {', '.join(selected_amenities) or 'None selected'}",
        "",
        "RESULTS",
        "ML classification: " + (
            "Good Investment — affordability label"
            if result["prediction"] == 1
            else "Does not meet affordability label"
        ),
        f"Uncalibrated label probability: {result['probability']:.4%}",
        f"Price per sq ft: INR {result['price_per_sqft']:,.2f}",
        f"Training benchmark per sq ft: INR {result['benchmark']:,.2f}",
        f"Overall benchmark fallback used: {result['fallback']}",
        "Direct affordability rule: " + (
            "Meets rule" if result["rule_result"] else "Does not meet rule"
        ),
        f"ML five-year price: INR {result['projected_price']:,.2f} lakh",
        f"Direct formula price: INR {result['formula_price']:,.2f} lakh",
        "",
        "ASSUMPTIONS AND LIMITATIONS",
        f"Annual growth assumption: {GROWTH:.0%}; horizon: {YEARS} years.",
        "Affordability is relative to a training-derived group median.",
        "The dataset contains no observed future prices or realised returns.",
        "Model probabilities are not probabilities of profit.",
        "Projections exclude rent, taxes, fees and inflation.",
        "This is an educational demonstration, not a validated market forecast."
    ]

    return "\n".join(lines)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.title("🏡 EstateScope")
    st.caption("REAL ESTATE INVESTMENT ADVISOR")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Property Advisor",
            "Explore Properties",
            "Compare Properties",
            "Model Insights"
        ]
    )

    st.divider()
    st.write("**Project dataset**")
    st.write(f"{len(properties):,} properties")
    st.write(f"{properties['State'].nunique()} states")
    st.write(f"{properties['City'].nunique()} cities")
    st.caption(f"Property ages reference {REFERENCE_YEAR}.")

    st.divider()
    st.caption(
        "Educational analysis using a constructed affordability "
        "rule and assumed growth. No guaranteed investment returns."
    )


# =========================================================
# Overview
# =========================================================

if page == "Overview":
    hero(
        "EXPLORE · COMPARE · UNDERSTAND",
        "A clearer view of property data",
        "Explore Indian property listings, compare affordability, "
        "and inspect transparent five-year growth scenarios."
    )

    a, b, c, d = st.columns(4)

    a.metric("Properties", f"{len(properties):,}")
    b.metric("Cities", properties["City"].nunique())
    c.metric(
        "Median listed price",
        f"₹{properties['Price_in_Lakhs'].median():,.2f} L"
    )
    d.metric(
        "Median price / sq ft",
        f"₹{properties['Price_per_SqFt'].median():,.0f}"
    )

    st.subheader("Choose your next step")

    left, middle, right = st.columns(3)

    with left:
        st.markdown("### 🔎 Analyze a property")
        st.write(
            "Enter property details to compare its price against "
            "a training-derived affordability benchmark."
        )

    with middle:
        st.markdown("### 🏘️ Explore the listings")
        st.write(
            "Filter by location, property type, BHK, budget and size. "
            "Download the matching records."
        )

    with right:
        st.markdown("### ⚖️ Compare two options")
        st.write(
            "Inspect two dataset listings side by side using the "
            "same units and assumptions."
        )

    st.divider()

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Property mix")

        counts = properties["Property_Type"].value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(
            counts.index,
            counts.values,
            color=["#52c7b8", "#e8bc70", "#6e9fdf"]
        )
        ax.set_xlabel("Number of properties")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        show_chart(fig)

    with chart_right:
        st.subheader("Listed price distribution")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            properties["Price_in_Lakhs"],
            bins=30,
            color="#52c7b8",
            edgecolor="#101c2f"
        )
        ax.set_xlabel("Price (₹ lakhs)")
        ax.set_ylabel("Properties")
        show_chart(fig)

    st.info(
        "The listed prices and property characteristics show limited "
        "relationships in this dataset. Treat its patterns as dataset "
        "findings rather than a description of the entire Indian market."
    )


# =========================================================
# Property advisor
# =========================================================

elif page == "Property Advisor":
    hero(
        "PROPERTY ADVISOR",
        "Understand your property scenario",
        "Submit a property, inspect the affordability benchmark, "
        "and compare illustrative growth assumptions."
    )

    example_names = {
        "Manual entry": None,
        "Demo: lower price per sq ft": 0,
        "Demo: higher price per sq ft": 1
    }

    example = st.selectbox(
        "Start with an example",
        list(example_names)
    )

    demo_row = None

    if example != "Manual entry":
        ranked = properties.sort_values("Price_per_SqFt")
        demo_row = (
            ranked.iloc[0]
            if example_names[example] == 0
            else ranked.iloc[-1]
        )

    states = sorted(properties["State"].unique().tolist())
    default_state = (
        str(demo_row["State"])
        if demo_row is not None
        else states[0]
    )

    location_left, location_right = st.columns(2)

    with location_left:
        state = st.selectbox(
            "State",
            states,
            index=states.index(default_state),
            key=f"advisor_state_{example}"
        )

    cities = sorted(
        properties.loc[
            properties["State"].eq(state), "City"
        ].unique().tolist()
    )

    default_city = (
        str(demo_row["City"])
        if demo_row is not None and str(demo_row["City"]) in cities
        else cities[0]
    )

    with location_right:
        city = st.selectbox(
            "City",
            cities,
            index=cities.index(default_city),
            key=f"advisor_city_{example}_{state}"
        )

    def default_value(column, fallback):
        return demo_row[column] if demo_row is not None else fallback

    with st.form(f"advisor_form_{example}"):
        st.subheader("Property details")
        left, middle, right = st.columns(3)

        with left:
            property_type = select_category(
                "Property type",
                "Property_Type",
                default_value("Property_Type", "Apartment")
            )

            bhk = st.number_input(
                "BHK", 1, 5,
                int(default_value("BHK", 2)),
                step=1
            )

            size = st.number_input(
                "Size (sq ft)", 500, 5000,
                int(default_value("Size_in_SqFt", 1500)),
                step=1
            )

            price = st.number_input(
                "Current price (₹ lakhs)", 10.0, 500.0,
                float(default_value("Price_in_Lakhs", 100.0)),
                step=1.0
            )

        with middle:
            age = st.number_input(
                f"Property age at {REFERENCE_YEAR}",
                2, 35,
                int(default_value("Age_of_Property", 10)),
                step=1
            )

            furnishing = select_category(
                "Furnishing",
                "Furnished_Status",
                default_value("Furnished_Status", "Furnished")
            )

            transport = select_category(
                "Public transport access",
                "Public_Transport_Accessibility",
                default_value("Public_Transport_Accessibility", "High")
            )

            schools = st.number_input(
                "Nearby schools", 1, 10,
                int(default_value("Nearby_Schools", 5)),
                step=1
            )

            hospitals = st.number_input(
                "Nearby hospitals", 1, 10,
                int(default_value("Nearby_Hospitals", 5)),
                step=1
            )

        with right:
            parking = select_category(
                "Parking", "Parking_Space",
                default_value("Parking_Space", "No")
            )

            security = select_category(
                "Security", "Security",
                default_value("Security", "No")
            )

            facing = select_category(
                "Facing", "Facing",
                default_value("Facing", "East")
            )

            owner_type = select_category(
                "Owner type", "Owner_Type",
                default_value("Owner_Type", "Broker")
            )

            availability = select_category(
                "Availability", "Availability_Status",
                default_value("Availability_Status", "Ready_to_Move")
            )

        default_amenities = (
            [
                amenity for amenity in AMENITIES
                if int(demo_row[f"Has_{amenity}"]) == 1
            ]
            if demo_row is not None
            else ["Garden"]
        )

        amenities = st.multiselect(
            "Amenities",
            AMENITIES,
            default=default_amenities
        )

        submitted = st.form_submit_button("Analyze property →")

    if submitted:
        values = {
            "State": state,
            "City": city,
            "Property_Type": property_type,
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Price_in_Lakhs": price,
            "Age_of_Property": age,
            "Nearby_Schools": schools,
            "Nearby_Hospitals": hospitals,
            "Furnished_Status": furnishing,
            "Public_Transport_Accessibility": transport,
            "Parking_Space": parking,
            "Security": security,
            "Facing": facing,
            "Owner_Type": owner_type,
            "Availability_Status": availability
        }

        for amenity in AMENITIES:
            values[f"Has_{amenity}"] = int(amenity in amenities)

        st.session_state["estate_analysis"] = analyze(values)

    if "estate_analysis" in st.session_state:
        result = st.session_state["estate_analysis"]
        saved = result["inputs"]

        st.divider()
        st.subheader("Your submitted analysis")

        st.caption(
            f"{saved['Property_Type']} · {saved['City']}, "
            f"{saved['State']} · {saved['BHK']} BHK · "
            f"{saved['Size_in_SqFt']:,} sq ft · "
            f"₹{saved['Price_in_Lakhs']:,.2f} lakh. "
            "Results retain the last submission; submit again after edits."
        )

        if result["prediction"] == 1:
            st.markdown(
                '<div class="status-good">'
                '<strong>✓ Good Investment — affordability label</strong>'
                '<br>The model predicts that this property meets '
                'the project’s relative-affordability rule.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-review">'
                '<strong>Review affordability</strong>'
                '<br>The model predicts that this property does not '
                'meet the project’s relative-affordability rule.</div>',
                unsafe_allow_html=True
            )

        a, b, c = st.columns(3)

        a.metric(
            f"ML price after {YEARS} years",
            f"₹{result['projected_price']:,.2f} L"
        )

        b.metric(
            "Current price per sq ft",
            f"₹{result['price_per_sqft']:,.2f}"
        )

        c.metric(
            "Affordability label probability",
            f"{result['probability']:.1%}"
        )

        st.caption(
            "Probability is an uncalibrated model score, rounded for "
            "display. It is not the probability of making a profit."
        )

        gap = (
            (result["price_per_sqft"] / result["benchmark"]) - 1
        ) * 100

        st.subheader("Affordability benchmark")

        st.write(
            f"The submitted price per sq ft is **{abs(gap):.1f}% "
            f"{'above' if gap > 0 else 'below' if gap < 0 else 'equal to'}** "
            "the training benchmark."
        )

        fig, ax = plt.subplots(figsize=(9, 3))

        ax.barh(
            ["Submitted property", "Training-group median"],
            [result["price_per_sqft"], result["benchmark"]],
            color=["#52c7b8", "#e8bc70"]
        )

        ax.set_xlabel("Price per square foot (₹)")
        ax.invert_yaxis()
        show_chart(fig)

        st.write(
            "**Direct rule:** "
            + (
                "Meets affordability rule"
                if result["rule_result"]
                else "Does not meet affordability rule"
            )
        )

        if result["fallback"]:
            st.caption(
                "The overall training median was used because "
                "a matching group benchmark was unavailable."
            )

        if result["prediction"] != int(result["rule_result"]):
            st.warning(
                "The model and direct rule disagree. The direct "
                "rule defines this project's classification target."
            )

        st.subheader("Five-year growth scenarios")

        st.caption(
            "Illustrative compound-growth calculations, not learned "
            "market forecasts. The saved model uses the 8% target."
        )

        scenario_data = pd.DataFrame({
            "Year": list(range(YEARS + 1))
        })

        for rate in [0.04, 0.08, 0.12]:
            scenario_data[f"{rate:.0%} annual assumption"] = (
                saved["Price_in_Lakhs"]
                * (1 + rate) ** scenario_data["Year"]
            )

        fig, ax = plt.subplots(figsize=(10, 4))

        for label, color in zip(
            scenario_data.columns[1:],
            ["#6e9fdf", "#52c7b8", "#e8bc70"]
        ):
            ax.plot(
                scenario_data["Year"],
                scenario_data[label],
                marker="o",
                linewidth=2,
                label=label,
                color=color
            )

        ax.set_xlabel("Years from the entered price")
        ax.set_ylabel("Projected price (₹ lakhs)")
        ax.set_xticks(range(YEARS + 1))
        ax.grid(alpha=0.3)
        ax.legend()
        show_chart(fig)

        scenario_summary = pd.DataFrame({
            "Annual assumption": ["4%", "8%", "12%"],
            "Price after five years (₹ lakhs)": [
                saved["Price_in_Lakhs"] * (1 + rate) ** YEARS
                for rate in [0.04, 0.08, 0.12]
            ]
        })

        st.dataframe(
            scenario_summary.round(2),
            hide_index=True
        )

        st.caption(
            "Scenario values exclude rental income, taxes, fees and "
            "inflation. Negative growth and losses are also possible."
        )

        st.download_button(
            "Download property analysis",
            data=analysis_report(result).encode("utf-8"),
            file_name="property_analysis.txt",
            mime="text/plain"
        )

        with st.expander("View submitted inputs"):
            st.dataframe(
                pd.DataFrame([saved]),
                hide_index=True
            )


# =========================================================
# Explore properties
# =========================================================

elif page == "Explore Properties":
    hero(
        "PROPERTY EXPLORER",
        "Find listings that fit your criteria",
        "Filter the dataset by location, budget, property type "
        "and size, then inspect the matching listings."
    )

    a, b, c = st.columns(3)

    with a:
        selected_state = st.selectbox(
            "State",
            ["All"] + sorted(properties["State"].unique())
        )

    state_data = (
        properties
        if selected_state == "All"
        else properties[
            properties["State"].eq(selected_state)
        ]
    )

    with b:
        selected_city = st.selectbox(
            "City",
            ["All"] + sorted(state_data["City"].unique()),
            key=f"explore_city_{selected_state}"
        )

    with c:
        selected_type = st.selectbox(
            "Property type",
            ["All"] + sorted(properties["Property_Type"].unique())
        )

    a, b, c = st.columns(3)

    with a:
        price_range = st.slider(
            "Price range (₹ lakhs)",
            10.0, 500.0, (10.0, 500.0), 1.0
        )

    with b:
        size_range = st.slider(
            "Size range (sq ft)",
            500, 5000, (500, 5000), 50
        )

    with c:
        bhk_choices = st.multiselect(
            "BHK",
            [1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5]
        )

    mask = (
        state_data["Price_in_Lakhs"].between(*price_range)
        & state_data["Size_in_SqFt"].between(*size_range)
        & state_data["BHK"].isin(bhk_choices)
    )

    if selected_city != "All":
        mask &= state_data["City"].eq(selected_city)

    if selected_type != "All":
        mask &= state_data["Property_Type"].eq(selected_type)

    filtered = state_data.loc[mask]

    if filtered.empty:
        st.warning(
            "No matching listings. Widen your filters or select more BHK values."
        )
    else:
        a, b, c = st.columns(3)

        a.metric("Matching listings", f"{len(filtered):,}")
        b.metric(
            "Median price",
            f"₹{filtered['Price_in_Lakhs'].median():,.2f} L"
        )
        c.metric(
            "Median price / sq ft",
            f"₹{filtered['Price_per_SqFt'].median():,.2f}"
        )

        sort_choice = st.selectbox(
            "Sort listings",
            [
                "Price: low to high",
                "Price: high to low",
                "Size: largest first",
                "Price per sq ft: low to high"
            ]
        )

        sorting = {
            "Price: low to high": ("Price_in_Lakhs", True),
            "Price: high to low": ("Price_in_Lakhs", False),
            "Size: largest first": ("Size_in_SqFt", False),
            "Price per sq ft: low to high": ("Price_per_SqFt", True)
        }

        sort_column, ascending = sorting[sort_choice]
        ordered = filtered.sort_values(sort_column, ascending=ascending)

        display_columns = [
            "ID", "State", "City", "Locality", "Property_Type",
            "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt", "Amenities_Count",
            "Furnished_Status", "Availability_Status"
        ]

        st.caption(
            "Preview shows the first 500 sorted matches. "
            "Use a property ID on the comparison page."
        )

        st.dataframe(
            ordered[display_columns].head(500),
            hide_index=True
        )

        st.download_button(
            "Download matching listings",
            data=ordered.to_csv(index=False).encode("utf-8"),
            file_name="matching_properties.csv",
            mime="text/csv"
        )

        left, right = st.columns(2)

        with left:
            st.subheader("Price distribution")

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(
                filtered["Price_in_Lakhs"],
                bins=25,
                color="#52c7b8",
                edgecolor="#101c2f"
            )
            ax.set_xlabel("Price (₹ lakhs)")
            ax.set_ylabel("Properties")
            show_chart(fig)

        with right:
            st.subheader("Top city averages")

            city_prices = (
                filtered.groupby(["State", "City"])["Price_in_Lakhs"]
                .mean()
                .nlargest(10)
                .sort_values()
            )

            labels = [
                f"{city_name}, {state_name}"
                for state_name, city_name in city_prices.index
            ]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(labels, city_prices.values, color="#e8bc70")
            ax.set_xlabel("Average price (₹ lakhs)")
            show_chart(fig)

        st.caption(
            "Charts use all matching records. These are comparisons "
            "of listed prices, not historical growth or investment returns."
        )


# =========================================================
# Compare properties
# =========================================================

elif page == "Compare Properties":
    hero(
        "SIDE-BY-SIDE COMPARISON",
        "Compare two property listings",
        "Use property IDs from the explorer to compare space, "
        "price, amenities and the same affordability rule."
    )

    st.caption(
        "The default examples use property IDs 1 and 2. "
        "Replace them with IDs from Explore Properties."
    )

    left, right = st.columns(2)

    with left:
        first_id = st.number_input(
            "Property A ID",
            min_value=int(properties["ID"].min()),
            max_value=int(properties["ID"].max()),
            value=int(properties.iloc[0]["ID"]),
            step=1
        )

    with right:
        second_id = st.number_input(
            "Property B ID",
            min_value=int(properties["ID"].min()),
            max_value=int(properties["ID"].max()),
            value=int(properties.iloc[1]["ID"]),
            step=1
        )

    first_match = properties[properties["ID"].eq(first_id)]
    second_match = properties[properties["ID"].eq(second_id)]

    if first_match.empty or second_match.empty:
        st.warning("One or both property IDs were not found.")
    else:
        if first_id == second_id:
            st.info("Choose different IDs to compare two distinct listings.")

        rows = [first_match.iloc[0], second_match.iloc[0]]
        comparison = {}

        for label, row in zip(["Property A", "Property B"], rows):
            benchmark, fallback = benchmark_for(
                row["State"],
                row["City"],
                row["Property_Type"]
            )

            comparison[label] = {
                "Property ID": int(row["ID"]),
                "State": row["State"],
                "City": row["City"],
                "Locality": row["Locality"],
                "Property type": row["Property_Type"],
                "BHK": int(row["BHK"]),
                "Size (sq ft)": int(row["Size_in_SqFt"]),
                "Current price (₹ lakhs)": round(float(row["Price_in_Lakhs"]), 2),
                "Price per sq ft (₹)": round(float(row["Price_per_SqFt"]), 2),
                "Training benchmark (₹ / sq ft)": round(benchmark, 2),
                "Direct affordability rule": (
                    "Meets rule"
                    if row["Price_per_SqFt"] <= benchmark
                    else "Does not meet rule"
                ),
                "Benchmark fallback used": "Yes" if fallback else "No",
                "Furnishing": row["Furnished_Status"],
                "Transport access": row["Public_Transport_Accessibility"],
                "Parking": row["Parking_Space"],
                "Security": row["Security"],
                "Amenities": row["Amenities"],
                "Availability": row["Availability_Status"],
                "Five-year formula projection (₹ lakhs)": round(
                    float(row["Price_in_Lakhs"]) * (1 + GROWTH) ** YEARS,
                    2
                )
            }

        comparison_table = pd.DataFrame(comparison)

        st.dataframe(comparison_table)

        st.download_button(
            "Download comparison",
            data=comparison_table.to_csv().encode("utf-8"),
            file_name="property_comparison.csv",
            mime="text/csv"
        )

        left, right = st.columns(2)

        with left:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Property A", "Property B"],
                [float(row["Price_in_Lakhs"]) for row in rows],
                color=["#52c7b8", "#e8bc70"]
            )
            ax.set_ylabel("Current price (₹ lakhs)")
            ax.set_title("Listed price")
            show_chart(fig)

        with right:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Property A", "Property B"],
                [float(row["Size_in_SqFt"]) for row in rows],
                color=["#52c7b8", "#e8bc70"]
            )
            ax.set_ylabel("Size (sq ft)")
            ax.set_title("Property size")
            show_chart(fig)

        st.caption(
            "No overall winner is assigned. Lower price or meeting "
            "the affordability rule does not establish better returns. "
            "Both five-year values use the same assumed growth rate."
        )


# =========================================================
# Model insights
# =========================================================

elif page == "Model Insights":
    hero(
        "TRANSPARENT MODEL RESULTS",
        "See what the models learned",
        "Review held-out test scores, validation comparisons "
        "and coefficient explanations."
    )

    class_metrics = metrics["classification"]
    reg_metrics = metrics["regression"]

    a, b, c, d = st.columns(4)

    a.metric("Classification accuracy", f"{class_metrics['Accuracy']:.2%}")
    b.metric("Classification F1", f"{class_metrics['F1']:.4f}")
    c.metric("Regression MAE", f"₹{reg_metrics['MAE_Lakhs'] * 100000:,.2f}")
    d.metric("Regression RMSE", f"₹{reg_metrics['RMSE_Lakhs'] * 100000:,.2f}")

    st.info(
        "Classification scores measure agreement with the constructed "
        "affordability labels. Regression errors measure agreement with "
        "the assumed growth formula. Neither evaluates realised returns."
    )

    st.subheader("Validation comparisons")

    left, right = st.columns(2)

    with left:
        st.write("**Classification**")
        path = BASE_DIR / "reports" / "classification_validation.csv"
        if path.exists():
            st.dataframe(pd.read_csv(path).round(4), hide_index=True)

    with right:
        st.write("**Regression — errors in lakhs**")
        path = BASE_DIR / "reports" / "regression_validation.csv"
        if path.exists():
            st.dataframe(pd.read_csv(path).round(6), hide_index=True)

    st.subheader("Model coefficient explanations")

    st.caption(
        "These are model weights after preprocessing, not causal "
        "effects or directly comparable universal importance scores."
    )

    left, right = st.columns(2)

    for column, pipeline, title, xlabel in [
        (
            left,
            classifier,
            "Classification coefficients",
            "Coefficient in log-odds"
        ),
        (
            right,
            regressor,
            "Regression coefficients",
            "Coefficient in target price (₹ lakhs)"
        )
    ]:
        with column:
            names = (
                pipeline.named_steps["preprocessing"]
                .get_feature_names_out()
            )

            coefficients = np.asarray(
                pipeline.named_steps["model"].coef_
            ).ravel()

            table = pd.DataFrame({
                "Feature": names,
                "Coefficient": coefficients
            })

            table["Absolute coefficient"] = table["Coefficient"].abs()

            top = (
                table.nlargest(10, "Absolute coefficient")
                .sort_values("Absolute coefficient")
            )

            readable_names = (
                top["Feature"]
                .str.replace("numeric__", "", regex=False)
                .str.replace("categorical__", "", regex=False)
                .str.replace("_", " ", regex=False)
            )

            colors = [
                "#52c7b8" if value >= 0 else "#e88e8e"
                for value in top["Coefficient"]
            ]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(readable_names, top["Coefficient"], color=colors)
            ax.axvline(0, color="#a2b2c8", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            show_chart(fig)

            with st.expander("All coefficients"):
                st.dataframe(
                    table.sort_values(
                        "Absolute coefficient",
                        ascending=False
                    ),
                    hide_index=True
                )

    st.caption(
        "Numerical inputs were standardized; categories were one-hot "
        "encoded. Correlated features and encoding affect coefficient "
        "interpretation. Current price dominates regression because "
        "the target is a constant multiple of current price."
    )

    with st.expander("MLflow experiment records"):
        path = BASE_DIR / "reports" / "mlflow_runs.csv"

        if path.exists():
            runs = pd.read_csv(path)
            wanted_columns = [
                "tags.mlflow.runName",
                "status",
                "metrics.accuracy",
                "metrics.f1",
                "metrics.mae_lakhs",
                "metrics.rmse_lakhs",
                "metrics.r2"
            ]

            available_columns = [
                column for column in wanted_columns
                if column in runs.columns
            ]

            st.dataframe(runs[available_columns], hide_index=True)

            st.caption(
                "Existing results were logged after training. Pipelines "
                "were saved as artifacts; this is not Model Registry registration."
            )
        else:
            st.write("The exported MLflow run summary is unavailable.")

    with st.expander("Methodology and limitations", expanded=True):
        st.markdown(
            f"""
- **Dataset:** {len(properties):,} properties; ages reference {REFERENCE_YEAR}.
- **Split:** 70% training, 15% validation and 15% test.
- **Preprocessing:** fitted using training data only.
- **Model selection:** based on validation scores.
- **Classification target:** price per sq ft at or below the
  training median for the same state, city and property type.
- **Regression target:** current price compounded at an assumed
  {GROWTH:.0%} annually for {YEARS} years.
- **Direct references:** the rule and formula reproduce their own
  targets exactly by construction.
- **Missing outcomes:** no observed future prices or realised returns.
- **Historical trends:** cannot be inferred from construction year.
- **Floor quality:** two floor columns were excluded because 46.52%
  of records had inconsistent floor information.
- **Evaluation scope:** full-dataset EDA preceded splitting;
  this is an educational holdout assessment, not an external test.
- **Probability:** uncalibrated and not a probability of profit.
"""
        )


st.divider()
st.caption(
    "EstateScope · Real Estate Investment Advisor · "
    "Educational analysis with explicit assumptions"
)