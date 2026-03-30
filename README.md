## Real Estate Price Prediction System

> Property value is estimated from patterns in data, not guesswork.

---

### Overview

This project predicts real estate property prices using machine learning. The solution was originally developed in a Jupyter Notebook and later modularized into a structured Python project using best practices such as separation of concerns, logging, and error handling.

A Streamlit web application is included to allow users to input property details and receive an estimated price in real time.

---

### Objectives

- Predict property prices using regression models  
- Convert notebook-based code into a modular Python project  
- Build an interactive web application using Streamlit  
- Apply logging and error handling for robustness  
- Prepare the project for GitHub publishing and deployment  

---

### Machine Learning Approach

- **Task Type:** Regression  
- **Target Variable:** `price`  

### Models Used
- Linear Regression  
- Random Forest Regressor (final model)

### Evaluation Metric
- Mean Absolute Error (MAE)

---

### Features Used

- Year Sold  
- Property Tax  
- Insurance  
- Number of Bedrooms  
- Number of Bathrooms  
- Living Area (sq ft)  
- Year Built  
- Lot Size  
- Basement Availability  
- Popular Area Indicator  
- Recession Indicator  
- Property Type  
- Property Age (derived feature)

---

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ByteWithShub/real-estate-price-prediction.git
cd real-estate-price-prediction
pip install -r requirements.txt
```

### Running Training and Application
python main.py
streamlit run app.py

Application Features: 
```
Predict property price based on input features
Display estimated price with interpretation
Show feature importance from the trained model
Perform what-if analysis to compare scenarios
```

### Output
The application provides:

- Estimated property price
- Price interpretation (low, mid-range, premium)
- Feature importance visualization
- Scenario comparison results


### Technologies Used
```
Python
Pandas
NumPy
Scikit-learn
Streamlit
Joblib
```

### Key Learning Outcomes
- Building regression models for real-world data
- Evaluating models using Mean Absolute Error
- Modularizing machine learning workflows
- Creating interactive ML applications
- Preparing projects for deployment and version control

### Author
```
Shubhangi Singh
```
