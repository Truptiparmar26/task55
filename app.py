import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Womens Clothing Prediction",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Womens Clothing Recommendation System")
st.write("Predict whether a customer will recommend a clothing product using Machine Learning.")

MODEL_DIR = "models"

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load(
            os.path.join(MODEL_DIR, "logistic_regression.pkl")
        ),
        "Random Forest": joblib.load(
            os.path.join(MODEL_DIR, "random_forest.pkl")
        ),
        "Gradient Boosting": joblib.load(
            os.path.join(MODEL_DIR, "gradient_boosting.pkl")
        )
    }
    return models

@st.cache_data
def load_data():
    file_path = "Womens Clothing E-Commerce Reviews.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        return df

    return pd.DataFrame()

try:
    models = load_models()
    df = load_data()

except Exception as e:
    st.error(f"Unable to load models or dataset: {e}")
    st.info("Please check that all model files are available inside the models folder.")
    st.stop()

st.success("Machine Learning Models Loaded Successfully!")

st.divider()

st.subheader("📊 Model Selection")

selected_model = st.selectbox(
    "Choose Machine Learning Model",
    list(models.keys())
)

st.divider()

st.subheader("📝 Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=30
    )

    rating = st.slider(
        "Product Rating",
        min_value=1,
        max_value=5,
        value=4
    )

    positive_feedback_count = st.number_input(
        "Positive Feedback Count",
        min_value=0,
        max_value=1000,
        value=2
    )

with col2:
    if not df.empty and "division_name" in df.columns:
        division_options = sorted(
            df["division_name"].dropna().astype(str).unique().tolist()
        )
    else:
        division_options = ["General", "General Petite", "Initmates"]

    division_name = st.selectbox(
        "Division Name",
        division_options
    )

    if not df.empty and "department_name" in df.columns:
        department_options = sorted(
            df["department_name"].dropna().astype(str).unique().tolist()
        )
    else:
        department_options = ["Tops", "Dresses", "Bottoms", "Intimate"]

    department_name = st.selectbox(
        "Department Name",
        department_options
    )

    if not df.empty and "class_name" in df.columns:
        class_options = sorted(
            df["class_name"].dropna().astype(str).unique().tolist()
        )
    else:
        class_options = ["Blouses", "Dresses", "Knits", "Pants"]

    class_name = st.selectbox(
        "Class Name",
        class_options
    )

st.divider()

if st.button("🔍 Predict Recommendation", type="primary", width="stretch"):

    input_data = pd.DataFrame([{
        "age": age,
        "rating": rating,
        "positive_feedback_count": positive_feedback_count,
        "division_name": division_name,
        "department_name": department_name,
        "class_name": class_name
    }])

    model = models[selected_model]

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("🎯 Prediction Result")

        if prediction == 1:
            st.success("Customer is likely to Recommend this Product! 👍")
        else:
            st.warning("Customer is unlikely to Recommend this Product.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Selected Model",
                selected_model
            )

        with col2:
            st.metric(
                "Prediction",
                "Recommended" if prediction == 1 else "Not Recommended"
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            predicted_index = classes.index(prediction)
            confidence = probabilities[predicted_index] * 100

            st.subheader("📈 Prediction Confidence")

            st.progress(float(confidence / 100))

            st.write(f"Confidence Score: **{confidence:.2f}%**")

            probability_data = pd.DataFrame({
                "Result": [
                    "Not Recommended",
                    "Recommended"
                ],
                "Probability": [
                    probabilities[classes.index(0)] * 100,
                    probabilities[classes.index(1)] * 100
                ]
            })

            st.bar_chart(
                probability_data.set_index("Result")
            )

        st.subheader("📋 Input Summary")
        st.dataframe(input_data, width="stretch")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.divider()

st.subheader("ℹ️ About This Application")

st.write("""
This application uses Machine Learning classification models
to predict whether a customer will recommend a clothing product.

Models used:
- Logistic Regression
- Random Forest
- Gradient Boosting

Dataset: Womens Clothing E-Commerce Reviews Dataset.

Target Variable: Recommended IND.
""")

st.caption("Womens Clothing: Trends and Analysis in the Fashion Industry")