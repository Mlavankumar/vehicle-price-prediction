# 🚗 Vehicle Price Prediction

This project builds a **Machine Learning model** that predicts the **price of vehicles** based on 
structured features such as make, model, year, mileage, cylinders, fuel type, transmission, body type, 
doors, and drivetrain.  

The goal is to help users estimate a fair price for vehicles using **data-driven insights**.  
The project uses a **RandomForestRegressor** with preprocessing pipelines for scaling and encoding.  

---

## 📊 Dataset
The dataset contains vehicle listings with the following fields:
- **make** – Vehicle manufacturer (e.g., Toyota, Honda, Ford)  
- **model** – Vehicle model (e.g., Camry, Civic)  
- **year** – Manufacturing year  
- **cylinders** – Engine cylinders  
- **fuel** – Fuel type (Gasoline, Diesel, Hybrid, Electric)  
- **mileage** – Distance traveled (km/miles)  
- **transmission** – Automatic / Manual  
- **body** – Body type (SUV, Sedan, Hatchback, etc.)  
- **doors** – Number of doors  
- **drivetrain** – FWD, RWD, AWD  
- **price** – Target variable (Vehicle price in currency units)  

---

## 🛠️ Skills Earned
By completing this project, you will gain hands-on experience with:
- ✅ **Data Preprocessing** – handling missing values, categorical encoding, scaling  
- ✅ **Feature Engineering** – selecting relevant features from structured data  
- ✅ **Machine Learning Pipelines** – using `ColumnTransformer` and `Pipeline` in scikit-learn  
- ✅ **Model Training** – RandomForestRegressor for regression tasks  
- ✅ **Model Evaluation** – Mean Absolute Error (MAE) and R² Score  
- ✅ **Model Persistence** – saving and loading trained models with `joblib`  
- ✅ **Practical Deployment Skills** – preparing prediction-ready pipelines  

---

## ⚙️ Requirements
Install the dependencies before running the project:


pandas
numpy
scikit-learn
joblib
jupyter


Install with:
pip install -r requirements.txt

How to Run
1. Clone the repository
git clone https://github.com/your-username/vehicle-price-prediction.git
cd vehicle-price-prediction

2. Open the Jupyter Notebook
jupyter notebook notebook/vehicle_price_prediction.ipynb

3. Train the Model
Run all notebook cells to:
Preprocess dataset
Train the RandomForest model
Evaluate results
Save the trained model (vehicle_price_model.pkl)

4. Predict Vehicle Price
Use the trained model for predictions:
import joblib
import pandas as pd

# Load model
model = joblib.load("vehicle_price_model.pkl")

# Example input
example = pd.DataFrame([{
    "make": "Toyota",
    "model": "Camry",
    "year": 2018,
    "cylinders": 4,
    "fuel": "Gasoline",
    "mileage": 45000,
    "transmission": "Automatic",
    "body": "Sedan",
    "doors": 4,
    "drivetrain": "FWD"
}])

# Predict
predicted_price = model.predict(example)
print(f"Predicted Vehicle Price: ₹{predicted_price[0]:,.2f}")
