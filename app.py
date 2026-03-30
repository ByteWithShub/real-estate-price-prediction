# Real Estate Price Prediction App

import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Real Estate Price Prediction",
    layout="wide"
)

model = joblib.load("models/real_estate_model.pkl")
columns = joblib.load("models/columns.pkl")

st.title("Real Estate Price Prediction")
st.write(
    "Enter the property details below to estimate the selling price based on the trained regression model."
)

st.markdown("---")

#Property Information

input_data = {}

st.subheader("Property Information")

col1, col2, col3 = st.columns(3)

with col1:
    input_data["year_sold"] = st.slider("Year Sold", 2000, 2025, 2015)
    input_data["beds"] = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=3)
    input_data["year_built"] = st.slider("Year Built", 1950, 2025, 2005)

with col2:
    input_data["baths"] = st.number_input("Number of Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    input_data["sqft"] = st.number_input("Living Area (sq ft)", min_value=0.0, max_value=10000.0, value=1500.0, step=50.0)
    input_data["lot_size"] = st.number_input("Lot Size", min_value=0.0, max_value=50000.0, value=5000.0, step=100.0)

with col3:
    input_data["property_tax"] = st.number_input("Annual Property Tax ($)", min_value=0.0, max_value=50000.0, value=3000.0, step=100.0)
    input_data["insurance"] = st.number_input("Annual Insurance ($)", min_value=0.0, max_value=10000.0, value=1500.0, step=50.0)

st.markdown("---")

#Additional Features

st.subheader("Additional Features")

col4, col5, col6 = st.columns(3)

with col4:
    basement = st.selectbox("Basement Available", ["No", "Yes"])
    input_data["basement"] = 1 if basement == "Yes" else 0

with col5:
    popular = st.selectbox("Located in a Popular Area", ["No", "Yes"])
    input_data["popular"] = 1 if popular == "Yes" else 0

with col6:
    recession = st.selectbox("Sold During a Recession", ["No", "Yes"])
    input_data["recession"] = 1 if recession == "Yes" else 0

st.markdown("---")

#Property Type
st.subheader("Property Type")

property_type = st.selectbox("Select Property Type", ["House", "Condo"])
input_data["property_type_Condo"] = 1 if property_type == "Condo" else 0

#Derived Feature: Property Age

input_data["property_age"] = input_data["year_sold"] - input_data["year_built"]

st.markdown("---")

#Prediction and Interpretation

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.subheader("Predicted Price")
    st.write(f"Estimated Property Price: ${prediction:,.2f}")

    st.subheader("Price Interpretation")

    if prediction < 200000:
        st.write(
            "Based on the model output, this property appears to be in the lower price range compared with other properties in the dataset."
        )
    elif prediction < 500000:
        st.write(
            "Based on the model output, this property appears to be in the mid-range price category."
        )
    else:
        st.write(
            "Based on the model output, this property appears to be in the premium price range."
        )

    st.subheader("Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))

    st.subheader("What-If Analysis")
    st.write(
        "Select one feature below, change its value, and compare the updated estimated price with the original prediction."
    )

    feature_to_change = st.selectbox("Select Feature to Modify", columns)
    current_value = float(input_data.get(feature_to_change, 0.0))

    new_value = st.number_input(
        "Enter New Value",
        value=current_value,
        key="what_if_value"
    )

    modified_input = input_data.copy()
    modified_input[feature_to_change] = new_value

    modified_df = pd.DataFrame([modified_input])
    modified_df = modified_df.reindex(columns=columns, fill_value=0)

    new_prediction = model.predict(modified_df)[0]
    difference = new_prediction - prediction

    col_a, col_b = st.columns(2)
    col_a.metric("Original Predicted Price", f"${prediction:,.2f}")
    col_b.metric("Updated Predicted Price", f"${new_prediction:,.2f}", f"{difference:,.2f}")

    if difference > 0:
        st.write(
            "In this scenario, changing the selected feature results in a higher estimated property price."
        )
    elif difference < 0:
        st.write(
            "In this scenario, changing the selected feature results in a lower estimated property price."
        )
    else:
        st.write(
            "In this scenario, the prediction remains unchanged."
        )

st.markdown("---")
st.caption("Built with Streamlit and Random Forest Regression.")