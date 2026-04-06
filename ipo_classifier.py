import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent

print("Loading data...")
df = pd.read_csv(BASE_DIR / 'nse_ipo_merged.csv')

# Drop rows where we don't have listing gains
df = df.dropna(subset=['Listing Day Gain (%)'])

# Target Variable: 1 if Listing Gain > 0 else 0
df['Profitable'] = (df['Listing Day Gain (%)'] > 0).astype(int)

# ── Feature Engineering ─────────────────────────────────────────────────────

# 1. Use existing IPO Year (already numeric int from clean_and_merge),
#    only re-derive if missing
if 'IPO Year' not in df.columns or df['IPO Year'].isna().all():
    df['IPO Start Date'] = pd.to_datetime(df['IPO Start Date'], format='%d-%b-%Y', errors='coerce')
    df['IPO Year'] = df['IPO Start Date'].dt.year
else:
    # Parse IPO Start Date for month extraction and sorting
    df['IPO Start Date'] = pd.to_datetime(df['IPO Start Date'], format='%d-%b-%Y', errors='coerce')

df['IPO Month'] = df['IPO Start Date'].dt.month

# 2. Mainboard vs SME flag (BE = Book Entry equity, treat as Mainboard)
df['Is_SME'] = df['Security Type'].apply(lambda x: 1 if x in ['SME', 'SM'] else 0)

# 3. Parse Issue Size to numeric (comes as strings like "Rs. 5,000 million")
#    Extract the numeric portion; rows that can't be parsed become NaN (handled by imputer)
df['Issue Size Numeric'] = pd.to_numeric(
    df['Issue Size'].astype(str).str.replace(',', '').str.extract(r'([\d.]+)', expand=False),
    errors='coerce'
)

# 4. Feature Definitions
#    - IPO Year and IPO Month are treated as NUMERIC (ordinal), not categorical.
#      This allows the model to generalize to unseen future years.
#    - Is_SME is already binary 0/1, treated as numeric.
numeric_features = [
    'Issue Price (₹)', 'Issue Size Numeric',
    'QIB Subscription (x)', 'NII Subscription (x)',
    'RII Subscription (x)', 'Total Subscription (x)',
    'IPO Year', 'IPO Month', 'Is_SME',
]

# Ensure numeric columns are actually numeric
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Preprocessing Pipeline ──────────────────────────────────────────────────

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
    ])

# ── Chronological Train/Test Split ──────────────────────────────────────────
# IMPORTANT: Sort FIRST, then create X and y, so iloc slicing is chronological.

df = df.sort_values('IPO Start Date').reset_index(drop=True)

X = df[numeric_features]
y = df['Profitable']

split_idx = int(len(df) * 0.8)  # 80% train, 20% test

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training on {len(X_train)} IPOs, Testing on {len(X_test)} IPOs.")
class_dist = y_test.value_counts(normalize=True).to_dict()
print(f"Test Set Class Distribution -> Profitable(1): {class_dist.get(1,0):.2%}, Loss/Flat(0): {class_dist.get(0,0):.2%}")

# ── Models ──────────────────────────────────────────────────────────────────

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, max_depth=4),
}

for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])

    # Train
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("-" * 50)
    print(f"🚀 Model: {name}")
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}\n")

    # Feature Importances (only for Tree based)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:7]  # Top 7

        print(f"Top 7 Features ({name}):")
        for i in indices:
            print(f"  {numeric_features[i]}: {importances[i]:.4f}")

    if name == 'Random Forest':
        print("\nExporting Random Forest model to ONNX...")
        try:
            from skl2onnx import to_onnx
            # Cast a sample dataframe to float32 to define the ONNX input schema correctly
            onx = to_onnx(clf, X_train[:1].astype(np.float32))
            with open("rf_ipo_classifier.onnx", "wb") as f:
                f.write(onx.SerializeToString())
            print("Model saved successfully as rf_ipo_classifier.onnx")
        except ImportError:
            print("skl2onnx not installed, skipping ONNX export.")
        except Exception as e:
            print(f"Failed to export ONNX: {e}")

