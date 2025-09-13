import streamlit as st
import joblib
import pandas as pd

# Load model, scaler, and features
model, scaler, features = joblib.load("attrition_model.pkl")

st.title("Employee Attrition Prediction 🚀")

st.write("Fill in the details below to predict whether an employee is likely to leave or stay.")

# --- User Inputs ---
Age = st.number_input("Age", 18, 60, 30)
DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
DistanceFromHome = st.number_input("Distance From Home (km)", 1, 30, 5)
Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
Gender = st.selectbox("Gender", ["Male", "Female"])
HourlyRate = st.number_input("Hourly Rate", 10, 100, 50)
JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
MonthlyRate = st.number_input("Monthly Rate", 1000, 30000, 10000)
NumCompaniesWorked = st.number_input("Number of Companies Worked", 0, 10, 1)
Over18 = st.selectbox("Over 18", ["Yes", "No"])
OverTime = st.selectbox("OverTime", ["Yes", "No"])
PercentSalaryHike = st.number_input("Percent Salary Hike", 0, 100, 15)
PerformanceRating = st.selectbox("Performance Rating", [1, 2, 3, 4])
RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 10)
TrainingTimesLastYear = st.number_input("Trainings Last Year", 0, 10, 2)
WorkLifeBalance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
YearsAtCompany = st.number_input("Years at Company", 0, 40, 5)
YearsInCurrentRole = st.number_input("Years in Current Role", 0, 20, 3)
YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 20, 1)
YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 20, 3)

# Engineered Features (if you had them during training)
YearsInOtherCompanies = st.number_input("Years in Other Companies", 0, 20, 2)
YearsSincePromotionRatio = st.number_input("Years Since Promotion Ratio", 0.0, 10.0, 1.0)
IncomePerYearOfExperience = st.number_input("Income per Year of Experience", 0.0, 100000.0, 5000.0)
AvgSatisfaction = st.number_input("Average Satisfaction", 1.0, 4.0, 3.0)
YearsPerJobLevel = st.number_input("Years per Job Level", 0.0, 20.0, 2.0)

# --- Encode categorical variables ---
Gender = 1 if Gender == "Male" else 0
OverTime = 1 if OverTime == "Yes" else 0
Over18 = 1 if Over18 == "Yes" else 0

# --- Construct input dataframe ---
data = pd.DataFrame([[
    Age, None, DailyRate, DistanceFromHome, Education,
    EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel,
    JobSatisfaction, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18,
    OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction,
    StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
    YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager,
    YearsInOtherCompanies, YearsSincePromotionRatio, IncomePerYearOfExperience,
    AvgSatisfaction, YearsPerJobLevel
]], columns=features)

# Note: 'Attrition' column (target) is excluded in features
if "Attrition" in data.columns:
    data = data.drop(columns=["Attrition"])

# --- Scale numeric features ---
numeric_features = ["Age", "DailyRate", "DistanceFromHome", "Education",
                    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
                    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
                    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
                    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
                    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
                    "YearsInCurrentRole", "YearsSinceLastPromotion",
                    "YearsWithCurrManager", "YearsInOtherCompanies",
                    "YearsSincePromotionRatio", "IncomePerYearOfExperience",
                    "AvgSatisfaction", "YearsPerJobLevel"]

data[numeric_features] = scaler.transform(data[numeric_features])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(data)
    result = "Attrition (Likely to Leave)" if prediction[0] == 1 else "No Attrition (Likely to Stay)"
    st.success(f"Prediction: {result}")
