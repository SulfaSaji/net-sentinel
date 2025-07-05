import pandas as pd
import joblib # Or 'pickle' if you used that to save your model/scalers
from sklearn.preprocessing import StandardScaler, LabelEncoder # Example scalers/encoders
import sys # For sys.exit if critical error
import os # For path checks

# --- IMPORTANT: Define paths to your saved model and preprocessing tools ---
# These paths should be relative to where your streamlit app (or live_detector.py) runs
MODEL_PATH = r'D:/intel/models/rf_model_smote.pkl' # Adjusted to match your specified model name
SCALER_PATH = r'D:/intel/models/scaler.joblib' # Adjusted to match your specified scaler name
ENCODER_PATH = r'D:/intel/models/label_encoder.joblib' # Adjusted to match your specified encoder name

# --- Your 77 Cleaned Feature Names (MUST EXACTLY MATCH model's training features) ---
FEATURE_NAMES = [
    'protocol', 'flow_duration', 'total_fwd_packets', 'total_backward_packets',
    'fwd_packets_length_total', 'bwd_packets_length_total', 'fwd_packet_length_max',
    'fwd_packet_length_min', 'fwd_packet_length_mean', 'fwd_packet_length_std',
    'bwd_packet_length_max', 'bwd_packet_length_min', 'bwd_packet_length_mean',
    'bwd_packet_length_std', 'flow_bytess', 'flow_packetss', 'flow_iat_mean',
    'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_total',
    'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
    'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max',
    'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags',
    'bwd_urg_flags', 'fwd_header_length', 'bwd_header_length',
    'fwd_packetss', 'bwd_packetss', 'packet_length_min', 'packet_length_max',
    'packet_length_mean', 'packet_length_std', 'packet_length_variance',
    'fin_flag_count', 'syn_flag_count',
    'rst_flag_count',
    'psh_flag_count',
    'ack_flag_count', 'urg_flag_count',
    'cwe_flag_count',
    'ece_flag_count',
    'downup_ratio', 'avg_packet_size', 'avg_fwd_segment_size',
    'avg_bwd_segment_size', 'fwd_avg_bytesbulk', 'fwd_avg_packetsbulk',
    'fwd_avg_bulk_rate', 'bwd_avg_bytesbulk', 'bwd_avg_packetsbulk',
    'bwd_avg_bulk_rate', 'subflow_fwd_packets', 'subflow_fwd_bytes',
    'subflow_bwd_packets', 'subflow_bwd_bytes', 'init_fwd_win_bytes',
    'init_bwd_win_bytes', 'fwd_act_data_packets', 'fwd_seg_size_min',
    'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean',
    'idle_std', 'idle_max', 'idle_min'
]

# --- Features to Exclude from Scaling ---
SCALED_EXCLUSION_LIST = [ 
    'protocol',
    'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
    'fin_flag_count', 'syn_flag_count',
    'rst_flag_count',
    'psh_flag_count',
    'ack_flag_count', 'urg_flag_count',
    'cwe_flag_count',
    'ece_flag_count',
    'downup_ratio',
    'fwd_avg_bytesbulk', 'fwd_avg_packetsbulk',
    'fwd_avg_bulk_rate', 'bwd_avg_bytesbulk', 'bwd_avg_packetsbulk',
    'bwd_avg_bulk_rate'
]

# --- Load the pre-trained model and preprocessing tools ---
model = None
scaler = None
label_encoder = None
label_encoder_classes_list = None # To store classes as a list for robust decoding

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error loading model file: {MODEL_PATH} not found. Make sure the path is correct and the model provider has placed the file.", file=sys.stderr)
    sys.exit(1) # Critical error, exit
except Exception as e:
    print(f"An unexpected error occurred loading model: {e}", file=sys.stderr)
    sys.exit(1) # Critical error, exit

try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print(f"Warning: Scaler file: {SCALER_PATH} not found. Ensure your model was not trained on scaled data, or provide the scaler.", file=sys.stderr)
    # This is a warning, not an exit, as some models might work without scaling, but accuracy will suffer.
except Exception as e:
    print(f"An unexpected error occurred loading scaler: {e}", file=sys.stderr)

try:
    label_encoder = joblib.load(ENCODER_PATH)
    label_encoder_classes_list = list(label_encoder.classes_) # Convert to list
    print("Label encoder loaded successfully.")
except FileNotFoundError:
    print(f"Warning: Label encoder file: {ENCODER_PATH} not found. Predictions will be numerical.", file=sys.stderr)
except Exception as e:
    print(f"An unexpected error occurred loading label encoder: {e}", file=sys.stderr)


def preprocess_data(df_input_raw, fitted_scaler):
    """
    Applies the same preprocessing steps as used for training data, including column renaming.
    IMPORTANT: This must mirror your training preprocessing exactly.
    - Column renaming
    - Feature selection
    - Scaling (using the *fitted* scaler)
    - Handling categorical features (e.g., OneHotEncoding, if not done by LabelEncoder for target)
    """
    df_processed = df_input_raw.copy()

    # --- Step 1: Rename DataFrame columns to match the snake_case FEATURE_NAMES ---
    column_rename_map = {}
    
    for original_col in df_processed.columns.tolist():
        # Exclude 'Label' from features, but allow it to be kept if needed elsewhere in df
        if original_col == 'Label':
            column_rename_map[original_col] = original_col # Keep label column name as is
            continue
        
        # Specific mappings for common CICIDS/UNSW-NB15 names that are not a simple lower().replace(' ', '_')
        if original_col == 'Flow Bytes/s':
            snake_case_version = 'flow_bytess'
        elif original_col == 'Flow Packets/s':
            snake_case_version = 'flow_packetss'
        elif original_col == 'Fwd Packets/s':
            snake_case_version = 'fwd_packetss'
        elif original_col == 'Bwd Packets/s':
            snake_case_version = 'bwd_packetss'
        elif original_col == 'Avg Packet Size':
            snake_case_version = 'avg_packet_size'
        elif original_col == 'Avg Fwd Segment Size':
            snake_case_version = 'avg_fwd_segment_size'
        elif original_col == 'Avg Bwd Segment Size':
            snake_case_version = 'avg_bwd_segment_size'
        elif original_col == 'Fwd Avg Bytes/Bulk':
            snake_case_version = 'fwd_avg_bytesbulk'
        elif original_col == 'Fwd Avg Packets/Bulk':
            snake_case_version = 'fwd_avg_packetsbulk'
        elif original_col == 'Fwd Avg Bulk Rate':
            snake_case_version = 'fwd_avg_bulk_rate'
        elif original_col == 'Bwd Avg Bytes/Bulk':
            snake_case_version = 'bwd_avg_bytesbulk'
        elif original_col == 'Bwd Avg Packets/Bulk':
            snake_case_version = 'bwd_avg_packetsbulk'
        elif original_col == 'Bwd Avg Bulk Rate':
            snake_case_version = 'bwd_avg_bulk_rate'
        elif original_col == 'Down/Up Ratio':
            snake_case_version = 'downup_ratio'
        elif original_col == 'FIN Flag Count':
            snake_case_version = 'fin_flag_count'
        elif original_col == 'SYN Flag Count':
            snake_case_version = 'syn_flag_count'
        elif original_col == 'RST Flag Count':
            snake_case_version = 'rst_flag_count'
        elif original_col == 'PSH Flag Count':
            snake_case_version = 'psh_flag_count'
        elif original_col == 'ACK Flag Count':
            snake_case_version = 'ack_flag_count'
        elif original_col == 'URG Flag Count':
            snake_case_version = 'urg_flag_count'
        elif original_col == 'CWE Flag Count':
            snake_case_version = 'cwe_flag_count'
        elif original_col == 'ECE Flag Count':
            snake_case_version = 'ece_flag_count'
        elif original_col == 'Init Fwd Win Bytes':
            snake_case_version = 'init_fwd_win_bytes'
        elif original_col == 'Init Bwd Win Bytes':
            snake_case_version = 'init_bwd_win_bytes'
        elif original_col == 'Fwd Act Data Packets':
            snake_case_version = 'fwd_act_data_packets'
        elif original_col == 'Fwd Seg Size Min':
            snake_case_version = 'fwd_seg_size_min'
        else:
            # General conversion for other columns (e.g., 'Flow Duration' -> 'flow_duration')
            # Remove any non-alphanumeric characters or special symbols from the name
            snake_case_version = original_col.lower()
            snake_case_version = ''.join(c if c.isalnum() else '_' for c in snake_case_version).strip('_')
            snake_case_version = '_'.join(filter(None, snake_case_version.split('_'))) # Remove multiple underscores

        # Only add to map if the target snake_case_version is in our expected FEATURE_NAMES
        if snake_case_version in FEATURE_NAMES:
            column_rename_map[original_col] = snake_case_version
        else:
            # If a column doesn't map to a feature, it will not be renamed and will be dropped later.
            pass

    df_processed.rename(columns=column_rename_map, inplace=True)
    print("DataFrame columns renamed to match expected feature names.")

    # --- Step 2: Validate and Select Features ---
    # Ensure all required features are in the DataFrame AFTER renaming
    missing_features = [f for f in FEATURE_NAMES if f not in df_processed.columns]
    if missing_features:
        raise ValueError(f"Missing required features in DataFrame AFTER renaming: {missing_features}. Please ensure your CSV's original column names correctly map to FEATURE_NAMES. Current columns: {list(df_processed.columns)}")

    # Select only the features needed for the model, in the correct order
    df_for_prediction = df_processed[FEATURE_NAMES].copy()

    # --- Step 3: Apply Scaling ---
    numerical_cols_for_scaling = [col for col in FEATURE_NAMES if col not in SCALED_EXCLUSION_LIST] # Changed variable name

    if fitted_scaler:
        try:
            # Ensure numerical columns are correctly typed before scaling
            for col in numerical_cols_for_scaling:
                df_for_prediction[col] = pd.to_numeric(df_for_prediction[col], errors='coerce').fillna(0)

            # Apply scaler.transform on the DataFrame, which preserves column names
            df_for_prediction[numerical_cols_for_scaling] = fitted_scaler.transform(df_for_prediction[numerical_cols_for_scaling])
            print("Features successfully scaled.")
        except Exception as e:
            raise RuntimeError(f"Error applying scaler: {e}. Check scaler compatibility and feature types/order.")
    else:
        print("Warning: Scaler not loaded. Features will not be scaled. Predictions may be inaccurate.")

    return df_for_prediction # This should be the X ready for prediction

def predict_intrusion(input_df_raw):
    """
    Takes raw input DataFrame, preprocesses it, and returns predictions.
    """
    if model is None:
        return pd.Series(["Error: Model not loaded."], index=input_df_raw.index)

    # 1. Preprocess the input data using the *same* steps and fitted scaler/encoder
    try:
        processed_df = preprocess_data(input_df_raw, scaler)

        # 2. Make predictions
        predictions = model.predict(processed_df)

        # 3. If your target was label encoded, decode it back for human readability
        if label_encoder_classes_list: # Use the list for robust decoding
            decoded_predictions = []
            for pred_val in predictions:
                pred_val_int = int(pred_val) # Ensure integer for indexing
                if 0 <= pred_val_int < len(label_encoder_classes_list):
                    decoded_predictions.append(label_encoder_classes_list[pred_val_int])
                else:
                    decoded_predictions.append(f"UNKNOWN_PREDICTED_LABEL_{pred_val_int}")
        else:
            decoded_predictions = predictions

        return pd.Series(decoded_predictions, index=input_df_raw.index)

    except ValueError as ve:
        return pd.Series([f"Preprocessing Error: {ve}"], index=input_df_raw.index)
    except RuntimeError as re:
        return pd.Series([f"Preprocessing/Scaling Error: {re}"], index=input_df_raw.index)
    except Exception as e:
        return pd.Series([f"Prediction Error: {e}"], index=input_df_raw.index)

# Example usage (for testing this file directly, not for Streamlit)
if __name__ == '__main__':
    # Create a dummy DataFrame that mimics your actual input CSV structure
    # Make sure column names and data types match what your model expects before preprocessing
    # This dummy data should use the *original* column names from your CSV, e.g., "Flow Bytes/s"
    dummy_data = {
        'Flow Duration': [0, 10, 20],
        'Protocol': [6, 17, 1], # Numerical protocol as per your model
        'Flow Bytes/s': [100.0, 500.0, 10.0],
        'FIN Flag Count': [0, 1, 0],
        'SYN Flag Count': [1, 0, 0],
        'RST Flag Count': [0, 0, 1],
        'PSH Flag Count': [1, 1, 0],
        'ACK Flag Count': [1, 0, 0],
        'URG Flag Count': [0, 0, 0],
        'CWE Flag Count': [0, 0, 0],
        'ECE Flag Count': [0, 0, 0],
        'Down/Up Ratio': [0.5, 0.0, 2.0],
        'Avg Packet Size': [150.0, 250.0, 30.0],
        'Avg Fwd Segment Size': [100.0, 500.0, 10.0],
        'Avg Bwd Segment Size': [200.0, 0.0, 20.0],
        'Fwd Avg Bytes/Bulk': [0.0, 0.0, 0.0],
        'Fwd Avg Packets/Bulk': [0.0, 0.0, 0.0],
        'Fwd Avg Bulk Rate': [0.0, 0.0, 0.0],
        'Bwd Avg Bytes/Bulk': [0.0, 0.0, 0.0],
        'Bwd Avg Packets/Bulk': [0.0, 0.0, 0.0],
        'Bwd Avg Bulk Rate': [0.0, 0.0, 0.0],
        'total_fwd_packets': [1, 5, 2],
        'total_backward_packets': [1, 0, 1],
        'fwd_packets_length_total': [100, 500, 20],
        'bwd_packets_length_total': [200, 0, 40],
        'fwd_packet_length_max': [100, 500, 20],
        'fwd_packet_length_min': [100, 100, 10],
        'fwd_packet_length_mean': [100, 250, 15],
        'fwd_packet_length_std': [0, 100, 5],
        'bwd_packet_length_max': [200, 0, 40],
        'bwd_packet_length_min': [200, 0, 20],
        'bwd_packet_length_mean': [200, 0, 30],
        'bwd_packet_length_std': [0, 0, 10],
        'flow_packetss': [1, 5, 2],
        'flow_iat_mean': [0, 2, 5],
        'flow_iat_std': [0, 1, 2],
        'flow_iat_max': [0, 5, 10],
        'flow_iat_min': [0, 1, 2],
        'fwd_iat_total': [0, 8, 10],
        'fwd_iat_mean': [0, 2, 5],
        'fwd_iat_std': [0, 1, 2],
        'fwd_iat_max': [0, 5, 10],
        'fwd_iat_min': [0, 1, 2],
        'bwd_iat_total': [0, 0, 5],
        'bwd_iat_mean': [0, 0, 5],
        'bwd_iat_std': [0, 0, 0],
        'bwd_iat_max': [0, 0, 5],
        'bwd_iat_min': [0, 0, 5],
        'fwd_header_length': [40, 120, 80],
        'bwd_header_length': [40, 0, 40],
        'fwd_packetss': [1, 5, 2],
        'bwd_packetss': [1, 0, 1],
        'packet_length_min': [100, 100, 10],
        'packet_length_max': [200, 500, 40],
        'packet_length_mean': [150, 250, 25],
        'packet_length_std': [50, 150, 10],
        'packet_length_variance': [2500, 22500, 100],
        'subflow_fwd_packets': [1, 5, 2],
        'subflow_fwd_bytes': [100, 500, 20],
        'subflow_bwd_packets': [1, 0, 1],
        'subflow_bwd_bytes': [200, 0, 40],
        'init_fwd_win_bytes': [65535, 65535, 65535],
        'init_bwd_win_bytes': [65535, -1, 65535],
        'fwd_act_data_packets': [1, 5, 2],
        'fwd_seg_size_min': [100, 100, 10],
        'active_mean': [0, 10, 20],
        'active_std': [0, 0, 0],
        'active_max': [0, 10, 20],
        'active_min': [0, 10, 20],
        'idle_mean': [0, 0, 0],
        'idle_std': [0, 0, 0],
        'idle_max': [0, 0, 0],
        'idle_min': [0, 0, 0],
        # Example of a 'Label' column, though it won't be used for prediction input
        'Label': ['NORMAL', 'DoS', 'Probe']
    }
    dummy_df = pd.DataFrame(dummy_data)

    print("--- Dummy Input Data (Original Column Names) ---")
    print(dummy_df.head())

    # To test this file directly, you would need dummy .pkl/.joblib files
    # in the 'model/' directory, mimicking your actual saved model, scaler, and encoder.
    # If you don't have them, the loading will fail and the script will exit.

    # predictions = predict_intrusion(dummy_df)
    # print("\n--- Dummy Predictions ---")
    # print(predictions)
