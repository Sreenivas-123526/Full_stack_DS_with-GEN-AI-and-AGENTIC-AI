import streamlit as st
import pandas as pd
import pickle

# Page Title
st.set_page_config(page_title="USA Housing Price Prediction", layout="wide")

st.title("🏠 USA Housing Price Prediction")

# Available Models
model_names = [
    'LinearRegression',
    'RobustRegression',
    'RidgeRegression',
    'LassoRegression',
    'ElasticNet',
    'PolynomilaRegression',
    'SGDRegressor',
    'ANN',
    'RandomForest',
    'SVM',
    'LGBM',
    'XGBoost',
    'KNN'
]

# Load Models
models = {}

for name in model_names:
    try:
        with open(f"{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    except:
        st.warning(f"{name}.pkl not found")

# Sidebar
st.sidebar.header("Select Model")

selected_model = st.sidebar.selectbox(
    "Choose Regression Model",
    model_names
)

# Input Fields
st.subheader("Enter House Details")

income = st.number_input(
    "Avg. Area Income",
    min_value=0.0,
    value=50000.0
)

house_age = st.number_input(
    "Avg. Area House Age",
    min_value=0.0,
    value=5.0
)

rooms = st.number_input(
    "Avg. Area Number of Rooms",
    min_value=0.0,
    value=7.0
)

bedrooms = st.number_input(
    "Avg. Area Number of Bedrooms",
    min_value=0.0,
    value=4.0
)

population = st.number_input(
    "Area Population",
    min_value=0.0,
    value=30000.0
)

# Prediction Button
if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        'Avg. Area Income':[income],
        'Avg. Area House Age':[house_age],
        'Avg. Area Number of Rooms':[rooms],
        'Avg. Area Number of Bedrooms':[bedrooms],
        'Area Population':[population]
    })

    model = models[selected_model]

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted House Price using {selected_model}: ${prediction:,.2f}"
    )

# Model Evaluation Results
st.subheader("Model Evaluation Results")

try:
    results_df = pd.read_csv(
        "model_evaluation_results.csv"
    )

    st.dataframe(results_df)

except:
    st.warning(
        "model_evaluation_results.csv not found"
    )