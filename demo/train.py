import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ALL_Cranial ILD data.xlsx"
OUT_PATH = Path(__file__).resolve().parents[1] / "models" / "sex_classifier.joblib"

SHEETS = ["AFRF", "AFRM", "ASNF", "ASNM", "EURF", "EURM"]

def load_all():
    parts = []
    for s in SHEETS:
        df = pd.read_excel(DATA_PATH, sheet_name=s)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        df = df.dropna(axis=1, how="all")
        df["ancestry"] = s[:3]
        parts.append(df)
    all_df = pd.concat(parts, ignore_index=True)
    all_df = all_df.dropna(subset=["Sex"])
    all_df["Sex"] = all_df["Sex"].astype(str).str.upper().str.strip()
    all_df = all_df[all_df["Sex"].isin(["F", "M"])]
    return all_df

def main():
    df = load_all()
    y = (df["Sex"] == "M").astype(int).to_numpy()
    ancestry = df["ancestry"].astype(str).to_numpy()

    feature_cols = [c for c in df.columns if c not in ["Sex", "ancestry"]]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    strat = np.array([f"{a}_{yy}" for a, yy in zip(ancestry, y)])

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []
    for tr, te in skf.split(X, strat):
        pipe.fit(X[tr], y[tr])
        p = pipe.predict(X[te])
        proba = pipe.predict_proba(X[te])[:, 1]
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p))
        aucs.append(roc_auc_score(y[te], proba))

    pipe.fit(X, y)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": pipe, "feature_cols": feature_cols},
        OUT_PATH
    )

    print("Saved:", OUT_PATH)
    print("CV accuracy:", float(np.mean(accs)))
    print("CV F1:", float(np.mean(f1s)))
    print("CV ROC-AUC:", float(np.mean(aucs)))
    print("n:", len(df), "features:", len(feature_cols))

if __name__ == "__main__":
    main()
