import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train(data_path: str = os.path.join("data", "laptop_data.csv")) -> None:
    df = pd.read_csv(data_path)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    df = df.dropna(subset=["Price"]).copy()

    df["Ram"] = df["Ram"].astype(str).str.replace("GB", "", regex=False).astype(int)
    df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).astype(float)

    # Touchscreen & IPS extracted from ScreenResolution
    df["Touchscreen"] = df["ScreenResolution"].astype(str).apply(lambda x: 1 if "Touchscreen" in x else 0)
    df["IPS"] = df["ScreenResolution"].astype(str).apply(lambda x: 1 if "IPS" in x else 0)

    df["Cpu Brand"] = df["Cpu"].astype(str).apply(lambda x: x.split()[0] if len(x.split()) else "Unknown")
    df["Gpu Brand"] = df["Gpu"].astype(str).apply(lambda x: x.split()[0] if len(x.split()) else "Unknown")

    df = df[[
        "Company",
        "TypeName",
        "Ram",
        "Weight",
        "Touchscreen",
        "IPS",
        "Cpu Brand",
        "Gpu Brand",
        "OpSys",
        "Price",
    ]].copy()

    df["Price"] = np.log(df["Price"])

    X = df.drop(columns=["Price"])
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["Company", "TypeName", "Cpu Brand", "Gpu Brand", "OpSys"],
            )
        ],
        remainder="passthrough",
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    print(f"Model R2 (log-price): {score:.4f}")

    os.makedirs("model", exist_ok=True)
    pickle.dump(pipe, open(os.path.join("model", "laptop_model.pkl"), "wb"))
    pickle.dump(df, open(os.path.join("model", "df.pkl"), "wb"))


if __name__ == "__main__":
    train()
