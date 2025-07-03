import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import sys

# --- Configuration Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT_DIR = SCRIPT_DIR

DATASET_RELATIVE_PATH = os.path.join('processed_data', 'final_netshield_cleaned_scaled_dataset.parquet')
DATASET_PATH = os.path.join(PROJECT_ROOT_DIR, DATASET_RELATIVE_PATH)

MODEL_SAVE_RELATIVE_DIR = 'models'
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT_DIR, MODEL_SAVE_RELATIVE_DIR)
MODEL_FILENAME = 'network_intrusion_model.pkl'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)

SCALER_SAVE_RELATIVE_DIR = 'models'
SCALER_SAVE_DIR = os.path.join(PROJECT_ROOT_DIR, SCALER_SAVE_RELATIVE_DIR)
SCALER_FILENAME = 'scaler.joblib'
SCALER_SAVE_PATH = os.path.join(SCALER_SAVE_DIR, SCALER_FILENAME)

LABEL_ENCODER_SAVE_RELATIVE_DIR = 'models'
LABEL_ENCODER_SAVE_DIR = os.path.join(PROJECT_ROOT_DIR, LABEL_ENCODER_SAVE_RELATIVE_DIR)
LABEL_ENCODER_FILENAME = 'label_encoder.joblib'
LABEL_ENCODER_SAVE_PATH = os.path.join(LABEL_ENCODER_SAVE_DIR, LABEL_ENCODER_FILENAME)


label_encoder = None
encoded_to_label = {}
NORMAL_ENCODED_VALUE = None

def load_dataset(path):
    print("--- Loading the Cleaned Dataset ---")
    print(f"Attempting to load dataset from: {path}")
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        print(f"Dataset loaded successfully from: {path}")
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at: {path}")
        print("Please double-check that the file name and the 'PROJECT_ROOT_DIR' are correct.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected problem occurred while loading the dataset:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("-" * 50)

def preprocess_labels(df, target_column='label'):
    global label_encoder, encoded_to_label, NORMAL_ENCODED_VALUE

    label_encoder = LabelEncoder()
    df[f'{target_column}_encoded'] = label_encoder.fit_transform(df[target_column])

    encoded_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    if 'NORMAL' in label_encoder.classes_:
        NORMAL_ENCODED_VALUE = label_encoder.transform(['NORMAL'])[0]
    else:
        print("WARNING: 'NORMAL' label not found in the dataset. NORMAL_ENCODED_VALUE not set.")
        NORMAL_ENCODED_VALUE = -1

    print("Mappings defined: encoded_to_label and NORMAL_ENCODED_VALUE.")

def split_data(df, target_column_encoded='label_encoded'):
    print("--- Dividing Data into Training and Test Sets ---")
    all_columns = df.columns.tolist()
    features_to_exclude = [target_column_encoded, target_column_encoded.replace('_encoded', '')]
    X_columns = [col for col in all_columns if col not in features_to_exclude]
    X = df[X_columns]
    y = df[target_column_encoded]

    print(f"\nShape of Features (X): {X.shape}")
    print(f"Shape of Target (y): {y.shape}")

    print("\n--- Performing Train-Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Data split into training and test sets successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("Error: 'imbalanced-learn' library not found. Please install it using: pip install imbalanced-learn")
        sys.exit(1)

    print("\n--- Applying SMOTE for Class Imbalance ---")
    class_counts = pd.Series(y_train).value_counts().sort_index()

    target_minority_size = 50000

    sampling_strategy = {}
    for cls, count in class_counts.items():
        if cls != NORMAL_ENCODED_VALUE:
            sampling_strategy[cls] = max(count, target_minority_size)
        else:
            sampling_strategy[cls] = count

    print("Proposed SMOTE sampling strategy:")
    print(sampling_strategy)

    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\n--- Class distribution AFTER SMOTE with custom strategy ---")
    print(pd.Series(y_train_resampled).value_counts().sort_index())
    return X_train_resampled, y_train_resampled

def train_model(X_train_resampled, y_train_resampled):
    print("\n--- Training RandomForestClassifier on SMOTE-resampled data ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print(f"Training data shape (resampled): {X_train_resampled.shape}")
    print(f"Training target shape (resampled): {y_train_resampled.shape}")

    rf_model.fit(X_train_resampled, y_train_resampled)
    print("\nRandomForestClassifier training on SMOTE-resampled data complete!")
    return rf_model

def evaluate_model(model, X_test, y_test):
    print("\n--- Evaluating the Model on the ORIGINAL Test Set (Multi-class) ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Set: {accuracy:.4f}")

    print("\nClassification Report (Multi-class):")
    target_names_multi = [encoded_to_label[i] for i in sorted(encoded_to_label.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names_multi, zero_division=0))

    print("\nConfusion Matrix (Multi-class):")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Binary (Malicious vs. Not Malicious) Classification Report ---")
    y_test_binary_eval = np.where(y_test == NORMAL_ENCODED_VALUE, 0, 1)
    y_pred_binary_eval = np.where(y_pred == NORMAL_ENCODED_VALUE, 0, 1)

    print(classification_report(y_test_binary_eval, y_pred_binary_eval, target_names=['Not Malicious (0)', 'Malicious (1)'], zero_division=0))

    print("\n--- Binary Confusion Matrix ---")
    print(confusion_matrix(y_test_binary_eval, y_pred_binary_eval))

    print("\n--- Investigating FALSE NEGATIVES (Actual Malicious, Predicted Not Malicious) ---")
    false_negatives_indices = np.where((y_test_binary_eval == 1) & (y_pred_binary_eval == 0))[0]

    if len(false_negatives_indices) > 0:
        print(f"\nTotal FALSE NEGATIVES: {len(false_negatives_indices)}")
        actual_labels_for_fns = y_test.iloc[false_negatives_indices]
        actual_fn_labels = [encoded_to_label[label] for label in actual_labels_for_fns]
        missed_attack_counts = pd.Series(actual_fn_labels).value_counts()
        print("\nDistribution of Missed Attack Types (False Negatives):")
        print(missed_attack_counts)
    else:
        print("No FALSE NEGATIVES found. Model is perfect at not missing attacks!")

def save_artifacts(model, label_encoder_obj, model_save_dir, model_filename, label_encoder_save_path):
    os.makedirs(model_save_dir, exist_ok=True)
    full_model_save_path = os.path.join(model_save_dir, model_filename)
    print(f"\nModel will be saved to: {full_model_save_path}")
    try:
        joblib.dump(model, full_model_save_path)
        print(f"Model '{model_filename}' successfully saved!")
    except NameError:
        print("Error: Model object not defined. Please ensure the model training was successful.")
    except Exception as e:
        print(f"Error saving model: {e}")

    print(f"Label encoder will be saved to: {label_encoder_save_path}")
    try:
        os.makedirs(os.path.dirname(label_encoder_save_path), exist_ok=True)
        joblib.dump(label_encoder_obj, label_encoder_save_path)
        print(f"Label encoder successfully saved!")
    except NameError:
        print("Error: 'label_encoder' object not defined. Please ensure label preprocessing was run.")
    except Exception as e:
        print(f"Error saving label encoder: {e}")


if __name__ == "__main__":
    print("--- Starting Network Intrusion Detection Model Training ---")

    df = load_dataset(DATASET_PATH)

    if df is not None:
        preprocess_labels(df)
    else:
        print("Data loading failed. Exiting.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    rf_model = train_model(X_train_resampled, y_train_resampled)

    evaluate_model(rf_model, X_test, y_test)

    save_artifacts(rf_model, label_encoder, MODEL_SAVE_DIR, MODEL_FILENAME, LABEL_ENCODER_SAVE_PATH)

    print("\n--- Network Intrusion Detection Model Training Complete ---")

