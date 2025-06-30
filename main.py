import streamlit as st
import numpy as np
import joblib
import pandas as pd


# Load model and scaler from disk
@st.cache_resource
def load_model_and_scaler():
    with open("titanic_model.pkl", "rb") as model_file:
        model = joblib.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = joblib.load(scaler_file)

    return model, scaler


model, scaler = load_model_and_scaler()

# UI Title
st.title("🚢 Titanic Survival Predictor")
st.write("Simulate whether you'd survive the Titanic based on your characteristics.")

# Input fields
sex = st.selectbox("Sex", ["male", "female"])
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 80, 30)
fare = st.slider("Fare Paid", 0.0, 500.0, 32.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Process inputs
sex_val = 0 if sex == "male" else 1
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0

# Scale Age and Fare
scaled_features = scaler.transform(pd.DataFrame([[age, fare]], columns=["Age", "Fare"]))
scaled_age = scaled_features[0][0]
scaled_fare = scaled_features[0][1]

# Construct input feature vector
input_features = np.array(
    [[pclass, sex_val, scaled_age, sibsp, parch, scaled_fare, embarked_C, embarked_Q]]
)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    st.subheader("🧾 Prediction Result:")
    if prediction == 1:
        st.success(
            f"You would have survived! 💪 (Survival Probability: {probability:.2%})"
        )
    else:
        st.error(
            f"You would not have survived. 😢 (Survival Probability: {probability:.2%})"
        )
