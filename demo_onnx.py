import onnxruntime as rt
import numpy as np
import pandas as pd
from pathlib import Path

# Path to the exported ONNX model
MODEL_PATH = "rf_ipo_classifier.onnx"
DATA_PATH = "nse_ipo_merged.csv"

def load_sample_data():
    """Loads a few IPOs from the dataset and applies the same minimal preprocessing needed before inference."""
    df = pd.read_csv(DATA_PATH)
    
    # Drop rows without listing gains to mimic training scenario, then pick top 3
    df = df.dropna(subset=['Listing Day Gain (%)']).reset_index(drop=True)
    sample_df = df.iloc[:3].copy()
    
    # Feature Engineering (must match ipo_classifier.py exactly)
    # 1. Date Features
    if 'IPO Year' not in sample_df.columns or sample_df['IPO Year'].isna().all():
        sample_df['IPO Start Date'] = pd.to_datetime(sample_df['IPO Start Date'], format='%d-%b-%Y', errors='coerce')
        sample_df['IPO Year'] = sample_df['IPO Start Date'].dt.year
    else:
        sample_df['IPO Start Date'] = pd.to_datetime(sample_df['IPO Start Date'], format='%d-%b-%Y', errors='coerce')

    sample_df['IPO Month'] = sample_df['IPO Start Date'].dt.month

    # 2. Mainboard vs SME flag
    sample_df['Is_SME'] = sample_df['Security Type'].apply(lambda x: 1 if x in ['SME', 'SM'] else 0)

    # 3. Numeric Issue Size
    sample_df['Issue Size Numeric'] = pd.to_numeric(
        sample_df['Issue Size'].astype(str).str.replace(',', '').str.extract(r'([\d.]+)', expand=False),
        errors='coerce'
    )

    numeric_features = [
        'Issue Price (₹)', 'Issue Size Numeric',
        'QIB Subscription (x)', 'NII Subscription (x)',
        'RII Subscription (x)', 'Total Subscription (x)',
        'IPO Year', 'IPO Month', 'Is_SME',
    ]

    return sample_df, numeric_features

def main():
    print(f"Loading ONNX Model: {MODEL_PATH}...")
    sess = rt.InferenceSession(MODEL_PATH)
    
    # Get the input names expected by the ONNX model (skl2onnx automatically sanitized the column names)
    onnx_inputs = sess.get_inputs()
    input_names = [i.name for i in onnx_inputs]
    
    # Load and prepare sample data
    df, feature_cols = load_sample_data()
    print(f"\nProcessing {len(df)} sample IPOs...")

    for idx, row in df.iterrows():
        company_name = row['Company Name']
        actual_gain = row['Listing Day Gain (%)']
        actual_class = "Profitable 📈" if actual_gain > 0 else "Flat/Loss 📉"

        # Prepare inputs dictionary mapping ONNX input names to actual values
        # They must be 2D float32 arrays -> shape [1, 1]
        inputs = {}
        for onnx_name, col_name in zip(input_names, feature_cols):
            # Grab single value, put it in 2D array, cast to float32
            val = row[col_name]
            # Handle NaN just in case (the ONNX pipeline has a SimpleImputer to handle nan)
            val = np.nan if pd.isna(val) else val
            inputs[onnx_name] = np.array([[val]], dtype=np.float32)

        # Run inference
        # sess.run(None, inputs) returns [predicted_labels, probabilities_dict]
        predictions = sess.run(None, inputs)
        
        pred_label = predictions[0][0] # 1 or 0
        pred_probs = predictions[1][0] # Dictionary: {0: prob_0, 1: prob_1}
        
        pred_class = "Profitable 📈" if pred_label == 1 else "Flat/Loss 📉"
        confidence = pred_probs.get(pred_label, 0) * 100
        
        print("-" * 50)
        print(f"🏢 Company: {company_name}")
        print(f"✅ Extracted Features: ")
        for f in feature_cols:
            print(f"   - {f}: {row[f]}")
        print(f"\n🔮 Prediction : {pred_class} (Confidence: {confidence:.2f}%)")
        print(f"📊 Actual     : {actual_class} (Gain: {actual_gain:+.2f}%)")

if __name__ == "__main__":
    main()
