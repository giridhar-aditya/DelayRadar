import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import shap
import matplotlib.pyplot as plt
import joblib

print("Loading dataset...")
df = pd.read_csv("flight.csv")

print("Dropping rows with missing values in important columns...")
df = df.dropna(subset=["DepDelayMinutes", "ArrDelayMinutes", "Distance", "Airline", "Origin", "Dest", "DepDel15"])
df['label'] = df['DepDel15'].astype(int)

print("Balancing dataset by downsampling majority class (label=0)...")
df_majority = df[df['label'] == 0]
df_minority = df[df['label'] == 1]
df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset size: {len(df_balanced)}")
print(df_balanced['label'].value_counts())

print("Preparing feature columns...")
categorical_cols = ["Airline", "Origin", "Dest"]
numerical_cols = ["Distance", "DepDelayMinutes", "ArrDelayMinutes", "AirTime", "CRSElapsedTime",
                  "ActualElapsedTime", "TaxiOut", "TaxiIn"]
feature_cols = categorical_cols + numerical_cols

X = df_balanced[feature_cols]
y = df_balanced['label']

print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

cat_features_indices = [X.columns.get_loc(col) for col in categorical_cols]

print("Creating CatBoost Pool objects for train and validation sets...")
train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
valid_pool = Pool(X_test, y_test, cat_features=cat_features_indices)

def objective(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0.1, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "task_type": "GPU",
        "devices": "0",
        "auto_class_weights": "Balanced",
        "eval_metric": "AUC",
        "verbose": 0,
        "early_stopping_rounds": 50
    }
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, verbose=0)
    preds = model.predict_proba(valid_pool)[:, 1]
    return roc_auc_score(y_test, preds)

print("Starting hyperparameter tuning with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best parameters found:")
print(study.best_params)

best_params = study.best_params
best_params.update({
    "task_type": "GPU",
    "devices": "0",
    "iterations": 1000,
    "auto_class_weights": "Balanced",
    "eval_metric": "AUC",
    "verbose": 100,
    "early_stopping_rounds": 50
})

print("Training final model with best parameters...")
final_model = CatBoostClassifier(**best_params)
final_model.fit(train_pool, eval_set=valid_pool)

print("Saving the trained model...")
final_model.save_model("flight_delay_catboost_model.cbm")
joblib.dump(final_model, "flight_delay_catboost_model.pkl")

print("Evaluating model on test set...")
y_pred = final_model.predict(valid_pool)
y_prob = final_model.predict_proba(valid_pool)[:, 1]

print("Classification report:")
print(classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_prob))

print("Plotting and saving confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("Calculating SHAP values for explainability...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

print("Generating SHAP summary plots...")
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Bar)")
plt.savefig("shap_feature_importance_bar.png", dpi=300, bbox_inches="tight")

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot")
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches="tight")

print("All done!")
