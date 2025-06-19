import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import shap
import matplotlib.pyplot as plt

print("Loading full dataset for evaluation...")
df = pd.read_csv("flight.csv")

print("Dropping rows with missing values...")
df = df.dropna(subset=["DepDelayMinutes", "ArrDelayMinutes", "Distance", "Airline", "Origin", "Dest", "DepDel15"])
df['label'] = df['DepDel15'].astype(int)

categorical_cols = ["Airline", "Origin", "Dest"]
numerical_cols = ["Distance", "DepDelayMinutes", "ArrDelayMinutes", "AirTime", "CRSElapsedTime",
                  "ActualElapsedTime", "TaxiOut", "TaxiIn"]
feature_cols = categorical_cols + numerical_cols

X = df[feature_cols]
y = df['label']

print("Loading trained CatBoost model...")
model = CatBoostClassifier()
model.load_model("flight_delay_catboost_model.cbm")

cat_features_indices = [X.columns.get_loc(col) for col in categorical_cols]
pool = Pool(X, y, cat_features=cat_features_indices)

print("Predicting on full dataset...")
y_pred = model.predict(pool)
y_prob = model.predict_proba(pool)[:, 1]

print("Classification report on full dataset:")
print(classification_report(y, y_pred))
print("AUC Score on full dataset:", roc_auc_score(y, y_prob))

print("Plotting confusion matrix...")
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Full Dataset")
plt.savefig("confusion_matrix_full.png", dpi=300)
plt.show()

print("Calculating SHAP values for full dataset...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("Generating SHAP summary plots for full dataset...")
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Bar) - Full Dataset")
plt.savefig("shap_feature_importance_bar_full.png", dpi=300, bbox_inches="tight")

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Summary Plot - Full Dataset")
plt.savefig("shap_summary_plot_full.png", dpi=300, bbox_inches="tight")

print("Evaluation complete.")
