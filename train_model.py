
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("insurance.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# 2. Data Preprocessing
# ==========================================

# Gender encoding
df["sex"] = df["sex"].map({
    "female": 1,
    "male": 0
})

# Smoker encoding
df["smoker"] = df["smoker"].map({
    "yes": 1,
    "no": 0
})


# Rename columns
df.rename(columns={
    "sex": "is_female",
    "smoker": "is_smoker"
}, inplace=True)


# ==========================================
# 3. BMI Category
# ==========================================

def bmi_category(bmi):

    if bmi < 18.5:
        return "Underweight"

    elif bmi < 25:
        return "Normal"

    elif bmi < 30:
        return "Overweight"

    else:
        return "Obese"


df["bmi_category"] = df["bmi"].apply(bmi_category)


# Convert BMI category into numerical columns
df = pd.get_dummies(
    df,
    columns=["bmi_category"],
    dtype=int
)


# Convert region into dummy variables
df = pd.get_dummies(
    df,
    columns=["region"],
    dtype=int
)


# ==========================================
# 4. Make sure required columns exist
# ==========================================

required_columns = [
    "age",
    "is_female",
    "bmi",
    "children",
    "is_smoker",
    "region_southeast",
    "bmi_category_Obese",
    "charges"
]

for column in required_columns:

    if column not in df.columns:
        df[column] = 0


# ==========================================
# 5. Select Features
# ==========================================

features = [
    "age",
    "is_female",
    "bmi",
    "children",
    "is_smoker",
    "region_southeast",
    "bmi_category_Obese"
]

X = df[features]

y = df["charges"]


# ==========================================
# 6. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 7. Scale Numerical Features
# ==========================================

numerical_columns = [
    "age",
    "bmi",
    "children"
]

scaler = StandardScaler()

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)


# ==========================================
# 8. Train Machine Learning Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 9. Prediction
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 10. Model Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ==========================================
# 11. Save Model
# ==========================================

joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)


print("\n==============================")
print("SUCCESS!")
print("==============================")

print("model.pkl created successfully!")
print("scaler.pkl created successfully!")
print("Training completed.")
