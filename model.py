import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv("NHIS Healthcare_claim_fraud.csv")   # keep your same dataset name

X = df.drop(columns=["FRAUD_TYPE"])   # keep same target column
y = df["FRAUD_TYPE"]

xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = xtrain.select_dtypes(include=["int64", "float64"]).columns
obj_cols = xtrain.select_dtypes(include=["object"]).columns
numeric_preprocessing = Pipeline(
    steps=[
        ("simple_imputer", SimpleImputer(strategy="mean")),
        ("standard_scaler", StandardScaler())
    ]
)

object_preprocessing = Pipeline(
    steps=[
        ("simple_imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ordinalEncoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ]
)

preprocessing = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocessing, num_cols),
        ("obj", object_preprocessing, obj_cols)
    ]
)

le = LabelEncoder()
ytrain_enc = le.fit_transform(ytrain)
ytest_enc = le.transform(ytest)

XG_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessing),
        ("model", XGBClassifier(classweight="balanced"))
    ]
)

XG_pipeline.fit(xtrain, ytrain_enc)
with open("model.pkl", "wb") as file:
    pickle.dump(XG_pipeline, file)

print("Model saved successfully as model.pkl")
