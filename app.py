import streamlit as st 
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")
import openai
import os
from dotenv import load_dotenv

# Load datasets
exercise_df = pd.read_csv("exercise.csv")
calories_df = pd.read_csv("calories.csv")
disease_df = pd.read_csv("disease_data.csv")  # Dataset for disease prediction

# Merge datasets on User_ID
df = pd.merge(exercise_df, calories_df, on="User_ID")
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Preprocess data for calorie prediction
X = df.drop(columns=["User_ID", "Calories"])
y = df["Calories"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Section 1: Algorithm Description
# Algorithm selection
st.title("Personal Fitness Tracking with AI")
algo_info = {
    "Random Forest": "An ensemble learning method using decision trees to improve accuracy.",
    "XGBoost": "An optimized gradient boosting algorithm with high predictive power."
}

model_choice = st.sidebar.selectbox("Choose Model", list(algo_info.keys()))
st.write(f"### {model_choice} Algorithm")
st.write(algo_info[model_choice])

# Train Model
if model_choice == "XGBoost":
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

# Section 2: User Information and Calorie Prediction
st.header("Calorie Burn Estimation")
user_data = {
    "Gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
    "Age": st.sidebar.number_input("Age", min_value=10, max_value=100, value=25),
    "Height": st.sidebar.number_input("Height (cm)", min_value=100, max_value=220, value=175),
    "Weight": st.sidebar.number_input("Weight (kg)", min_value=30, max_value=150, value=70),
    "Duration": st.sidebar.number_input("Workout Duration (min)", min_value=5, max_value=180, value=30),
    "Heart_Rate": st.sidebar.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=120),
    "Body_Temp": st.sidebar.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
}
user_data["Gender"] = 0 if user_data["Gender"] == "Male" else 1
input_df = pd.DataFrame([user_data])

prediction = model.predict(input_df)
calories_per_min = prediction[0] / user_data["Duration"]
st.write(f"Estimated Calories Burned: **{round(prediction[0], 2)} kcal**")
st.write(f"Calories per min: **{round(calories_per_min, 2)} kcal/min**")
for t in [30, 60, 90]:
    st.write(f"For {t} min: **{round(calories_per_min * t, 2)} kcal**")

# Section 3: Shopping Cart
st.header("Fitness Store - Add to Cart")
products = {"Protein Powder": 20, "Dumbbells": 30, "Yoga Mat": 15}
cart = []
for product, price in products.items():
    if st.checkbox(f"Add {product} - ${price}"):
        cart.append((product, price))

# Add clickable links
product_links = {"Protein Powder": "https://example.com/protein", "Dumbbells": "https://example.com/dumbbells", "Yoga Mat": "https://example.com/yogamat"}
st.write("### Product Links")
for product, link in product_links.items():
    st.markdown(f"[{product}]({link})")

total = sum([item[1] for item in cart])
st.write(f"Total Cart Value: **${total}**")

# SECTION 4: Disease Risk Prediction
st.header("Disease Risk Prediction")

disease_features = disease_df.drop(columns=["Disease"])
disease_labels = disease_df["Disease"].astype('category').cat.codes  # Convert categorical labels to numeric

disease_X_train, disease_X_test, disease_y_train, disease_y_test = train_test_split(
    disease_features, disease_labels, test_size=0.2, random_state=42
)

disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
disease_model.fit(disease_X_train, disease_y_train)

user_disease_input = np.random.rand(1, disease_features.shape[1])  # Placeholder user input
disease_pred_prob = disease_model.predict_proba(user_disease_input)[0]

diseases = disease_df["Disease"].unique()
for i, disease in enumerate(diseases):
    risk_percentage = round(disease_pred_prob[i] * 100, 2)
    st.write(f"**{disease} Risk:** {risk_percentage}%")
    st.progress(disease_pred_prob[i])

@st.cache_data
def load_data():
    df = pd.read_csv("exercise_dataset.csv")  # Use the uploaded dataset
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    return df

df = load_data()

# Section 5: Calories and Diet
st.subheader("🍽️ Calorie Tracker & Diet Planning")

@st.cache_data
def load_food_data():
    df = pd.read_csv("calories_food.csv")

    # Ensure the "Calories" column is cleaned and numeric
    df["Calories"] = df["Calories"].astype(str).str.replace(r"\D+", "", regex=True).astype(float)
    
    return df

food_df = load_food_data()

# Function to calculate daily calorie needs
def calculate_calories(gender, age, weight, height, activity_level):
    if gender == "Male":
        bmr = 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:
        bmr = 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)
    return bmr * activity_level

# User inputs
gender = st.selectbox("Select Gender", ["Male", "Female"])
age = st.number_input("Enter Age", min_value=10, max_value=100, value=25)
weight = st.number_input("Enter Weight (kg)", min_value=30, max_value=150, value=70)
height = st.number_input("Enter Height (cm)", min_value=100, max_value=220, value=175)
activity_level = st.selectbox("Select Activity Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"])

# Calculate and display daily calorie needs
activity_levels = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725, "Very Active": 1.9}
calories_needed = calculate_calories(gender, age, weight, height, activity_levels[activity_level])
st.write(f"Your estimated daily calorie requirement is **{round(calories_needed, 2)} kcal**")

# Display food suggestions
st.subheader("Suggested Foods for Your Diet")
suggested_foods = food_df[food_df["Calories"] <= calories_needed]
st.write(suggested_foods[["Food", "Calories"]])
