import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- IMPORTANT: Import your nids_core module ---
try:
    from utils import nids_core
except ImportError as e:
    st.error(f"Could not import nids_core.py. Make sure it's in the same directory and contains no syntax errors: {e}")
    st.stop()

# --- Streamlit App UI ---
st.set_page_config(page_title="NIDS Anomaly Detector", layout="centered")

st.title("🛡️ Network Intrusion Detection System (NIDS)")
st.markdown("Upload a CSV file containing network traffic data to detect anomalies and calculate accuracy.")

# File Uploader Widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.success("File successfully uploaded!")

    try:
        # Read the uploaded CSV file into a Pandas DataFrame
        input_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview:")
        st.dataframe(input_df.head())

        # --- Check for 'Label' column for accuracy calculation ---
        # We need the original 'Label' column to compare against predictions
        original_labels_exist = 'Label' in input_df.columns

        if original_labels_exist:
            st.info("Original 'Label' column found. Accuracy will be calculated.")
            # --- Standardize the 'Label' column from the uploaded CSV ---
            # This must mirror the standardization done on predictions
            # Ensure consistent mapping: Benign -> NORMAL, Infiltration -> INFILTRATION
            # Apply UPPER first, then specific replacements
            input_df['Label_Standardized'] = input_df['Label'].astype(str).str.upper()
            input_df['Label_Standardized'] = input_df['Label_Standardized'].replace({
                "BENIGN": "NORMAL",
                # Add other mappings here if needed, e.g.,
                # "BRUTEFORCE": "BRUTEFORCE", # If you had a different casing like "BruteForce"
                # "DOS": "DOS",
                # "WEB ATTACK": "WEB_ATTACK" # Example if you had spaces
            })
            # For "Infiltration", if it's already uppercase, no need for specific replace
            # If you had "Infiltrate" and wanted "INFILTRATION", you'd add:
            # input_df['Label_Standardized'] = input_df['Label_Standardized'].replace({"INFILTRATE": "INFILTRATION"})


        # Check if the nids_core module and its prediction function are ready
        if hasattr(nids_core, 'predict_intrusion') and callable(nids_core.predict_intrusion):
            st.info("Running anomaly detection... This may take a moment for large files.")
            
            # --- Perform Prediction ---
            # nids_core.predict_intrusion should handle model loading, preprocessing, and prediction
            # It should return already decoded/standardized predictions if label_encoder is used.
            predictions = nids_core.predict_intrusion(input_df.copy())

            st.subheader("Prediction Results:")
            
            if predictions.dtype == object and "Error:" in predictions.iloc[0]:
                st.error(f"Prediction failed: {predictions.iloc[0]}")
            else:
                # Add predictions back to the original input DataFrame for display
                input_df['Predicted_Anomaly'] = predictions

                # Display a summary of anomalies
                st.write("### Anomaly Counts:")
                anomaly_counts = input_df['Predicted_Anomaly'].value_counts()
                st.dataframe(anomaly_counts) # Use st.dataframe for better display of series

                # --- Calculate and Display Accuracy ---
                if original_labels_exist:
                    # Ensure the predicted anomalies are also in a consistent case for comparison
                    # nids_core should already return standardized predictions, but a final UPPER ensures it
                    input_df['Predicted_Anomaly_Standardized'] = input_df['Predicted_Anomaly'].astype(str).str.upper()
                    # Apply any specific replacements if nids_core's output needs further standardization
                    # For example, if nids_core returned "normal" instead of "NORMAL"
                    input_df['Predicted_Anomaly_Standardized'] = input_df['Predicted_Anomaly_Standardized'].replace({
                        # "NORMAL": "NORMAL", # Redundant if already uppercase
                        # "BRUTEFORCE": "BRUTEFORCE",
                    })

                    # Compare the standardized labels and predictions
                    correct_predictions_count = (input_df['Label_Standardized'] == input_df['Predicted_Anomaly_Standardized']).sum()
                    total_predictions_count = len(input_df)
                    
                    if total_predictions_count > 0:
                        accuracy = (correct_predictions_count / total_predictions_count) * 100
                        st.markdown("---")
                        st.subheader("Model Accuracy:")
                        st.metric(label="Overall Accuracy", value=f"{accuracy:.2f}%")
                        st.info(f"({correct_predictions_count} out of {total_predictions_count} predictions were correct)")
                    else:
                        st.warning("No data rows to calculate accuracy.")
                else:
                    st.warning("Cannot calculate accuracy: 'Label' column not found in the uploaded CSV.")

                st.markdown("---")
                st.subheader("Detailed Results (with Predictions):")
                # Display only relevant columns for detailed results, or the full df
                display_cols = input_df.columns.tolist()
                if 'Label_Standardized' in display_cols:
                    display_cols.remove('Label_Standardized')
                if 'Predicted_Anomaly_Standardized' in display_cols:
                    display_cols.remove('Predicted_Anomaly_Standardized')
                
                st.dataframe(input_df[display_cols])

                # Optional: Download results
                csv_output = input_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_output,
                    file_name="nids_predictions.csv",
                    mime="text/csv",
                )
        else:
            st.error("`predict_intrusion` function not found or not callable in `nids_core.py`. Please check its definition.")

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a file with data.")
    except Exception as e:
        st.error(f"An error occurred during file processing or prediction: {e}")
        st.warning("Please ensure your CSV file format matches the expected input for the model, and all necessary features are present. Also check the `server.maxMessageSize` config option if the file is large.")