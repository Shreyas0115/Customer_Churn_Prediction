"""
Customer Churn Prediction
- Input CSV with a churn label (guesses 'churn'/'Churn'/'Exited' if not provided).
- Numeric + categorical preprocessing via ColumnTransformer.
- Tries Logistic Regression, Random Forest, Gradient Boosting; selects best by ROC-AUC.

"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

def infer_target(df, target_col):
    if target_col is None:
        for guess in ["churn", "Churn", "Exited", "exit", "is_churn"]:
            if guess in df.columns:
                target_col = guess
                break
    if target_col is None or target_col not in df.columns:
        raise ValueError(f"Target column not found. Available: {list(df.columns)}")
    return target_col

def split_columns(df, target_col):
    X_cols = [c for c in df.columns if c != target_col]
    cat_cols = [c for c in X_cols if df[c].dtype == "object"]
    num_cols = [c for c in X_cols if c not in cat_cols]
    return num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )
    return preprocessor

def train_models(X_train, y_train, preprocessor):
    models = {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        "GradBoost": GradientBoostingClassifier()
    }
    fitted = {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted

def evaluate_models(models, X_test, y_test):
    scores = {}
    for name, pipe in models.items():
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            probas = pipe.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, probas)
        else:
            # Fallback via decision function if no predict_proba (not the case here)
            preds = pipe.predict(X_test)
            roc = np.nan
        scores[name] = roc
    return scores

def main(args):
    df = pd.read_csv(args.csv)
    target_col = infer_target(df, args.target_col)
    y = df[target_col].astype(int)
    num_cols, cat_cols = split_columns(df, target_col)
    X = df[num_cols + cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)
    models = train_models(X_train, y_train, preprocessor)
    scores = evaluate_models(models, X_test, y_test)

    # select best
    best_name = max(scores, key=lambda k: (scores[k] if scores[k] == scores[k] else -1))
    best_model = models[best_name]

    print("=== ROC-AUC by Model (test) ===")
    for name, sc in scores.items():
        print(f"{name}: {sc:.4f}")

    preds = best_model.predict(X_test)
    print(f"\n=== Best Model: {best_name} ===")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    # Optional: permutation importance for interpretability (may take time)
    try:
        if hasattr(best_model.named_steps["clf"], "feature_importances_") or isinstance(best_model.named_steps["clf"], GradientBoostingClassifier):
            r = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
            # Map back feature names from preprocessor
            ohe = best_model.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
            num_names = num_cols
            cat_names = list(ohe.get_feature_names_out(cat_cols))
            feat_names = list(num_names) + cat_names
            importances = sorted(zip(feat_names, r.importances_mean), key=lambda t: t[1], reverse=True)[:20]
            print("\nTop 20 Permutation Importances:")
            for name, val in importances:
                print(f"{name:40s} {val:.6f}")
    except Exception as e:
        print("Permutation importance skipped:", e)

    # Save model
    out_path = args.model_out
    meta = {
        "best_model": best_name,
        "target_col": target_col,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "test_size": args.test_size
    }
    joblib.dump({"pipeline": best_model, "meta": meta}, out_path)
    print(f"\nSaved model to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV with features and target.")
    parser.add_argument("--target_col", default=None, help="Target column name (default: auto-detect).")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--model_out", default="churn_model.joblib")
    args = parser.parse_args()
    main(args)
