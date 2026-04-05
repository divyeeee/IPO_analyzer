import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
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

# Feature Engineering
# 1. Date Features
df['IPO Start Date'] = pd.to_datetime(df['IPO Start Date'], format='%d-%b-%Y', errors='coerce')
df['IPO Month'] = df['IPO Start Date'].dt.month
df['IPO Year'] = df['IPO Start Date'].dt.year

# 2. Mainboard vs SME (SME has shown completely different dynamics)
df['Is_SME'] = df['Security Type'].apply(lambda x: 1 if x in ['SME', 'SM'] else 0)

# 3. Categorical & Numerical Feature Definitions
numeric_features = [
    'Issue Price (₹)', 'Issue Size', 
    'QIB Subscription (x)', 'NII Subscription (x)', 
    'RII Subscription (x)', 'Total Subscription (x)'
]

categorical_features = ['IPO Month', 'IPO Year', 'Is_SME']

# Fill potentially missing data using proper imputation later, but basic cleanup first
# Ensure numeric columns are actually numeric
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = df[numeric_features + categorical_features]
y = df['Profitable']

# Train / Test split - Since it's time series-like, a random split might look too good but 
# chronological split is preferred to test real world predictability.
# Sort by date
df = df.sort_values('IPO Start Date')
split_idx = int(len(df) * 0.8) # 80% train, 20% test

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training on {len(X_train)} IPOs, Testing on {len(X_test)} IPOs.")
class_dist = y_test.value_counts(normalize=True).to_dict()
print(f"Test Set Class Distribution -> Profitable(1): {class_dist.get(1,0):.2%}, Loss/Flat(0): {class_dist.get(0,0):.2%}")

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, max_depth=4)
}

results = {}

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
    
    # Feature Importances (only for Tree based)
    if hasattr(model, 'feature_importances_'):
        # Get feature names after one hot encoding
        ohe = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_names = ohe.get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(cat_names)
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:7] # Top 7
        
        print(f"Top 7 Features ({name}):")
        for i in indices:
            print(f"  {feature_names[i]}: {importances[i]:.4f}")

