# HR Attrition Analysis 🚀

Predict whether an employee is likely to leave the company using machine learning and Streamlit.

---

## 🧰 Features Used
The app uses **all features available in the dataset**, including:

- Age
- DailyRate
- DistanceFromHome
- Education
- EnvironmentSatisfaction
- Gender
- HourlyRate
- JobInvolvement
- JobLevel
- JobSatisfaction
- MonthlyIncome
- MonthlyRate
- NumCompaniesWorked
- Over18
- OverTime
- PercentSalaryHike
- PerformanceRating
- RelationshipSatisfaction
- StockOptionLevel
- TotalWorkingYears
- TrainingTimesLastYear
- WorkLifeBalance
- YearsAtCompany
- YearsInCurrentRole
- YearsSinceLastPromotion
- YearsWithCurrManager
- YearsInOtherCompanies
- YearsSincePromotionRatio
- IncomePerYearOfExperience
- AvgSatisfaction
- YearsPerJobLevel

> All features are required for accurate predictions.

---

## 📂 Files
- `app.py` – Streamlit app for predictions  
- `notebook.ipynb` – Jupyter notebook with data preprocessing, EDA, feature engineering, and model training  
- `attrition_model.pkl` – Saved trained machine learning model  
- `requirements.txt` – Python dependencies  

---

## 💻 How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/HR_Attrition_Prediction.git
cd HR_Attrition_Prediction
````

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**

```bash
streamlit run app.py
```

4. **Open the app** in your browser (usually at [http://localhost:8501](http://localhost:8501)).

---

## 🛠 Model Training

* The model uses all available features from the HR dataset.
* Preprocessing includes label encoding for categorical variables and scaling for numeric variables.
* Feature engineering creates additional meaningful metrics such as:

  * YearsSincePromotionRatio
  * IncomePerYearOfExperience
  * AvgSatisfaction
  * YearsPerJobLevel
* Machine learning model: `RandomForestClassifier` and `Logistic Regression` from scikit-learn.

---

## 📌 Notes

* All features must be provided for accurate predictions.
* Make sure `attrition_model.pkl` and `scaler.pkl` are in the same folder as `app.py`.
* The app is designed for demonstration and portfolio purposes.

---

## Author

Nishant Dorwal

```
