# Network Intrusion Detection System (NIDS)

-----

## 🚀 Project Overview

This project develops an **AI-driven Network Intrusion Detection System (NIDS)** designed to identify and classify various types of network attacks (e.g., DoS, DDoS, Web Attacks, Port Scans). It features a user-friendly Streamlit web interface, allowing users to upload CSV files for classification. Leveraging machine learning techniques, the system aims to enhance network security by detecting anomalies and malicious activities with high accuracy and efficiency, specifically focusing on minimizing false negatives for critical attack types. The application allows users to upload a dataset, get it classified, view classification accuracy, and download the classified results.

-----

## ✨ Features & Goals

The primary objectives and capabilities of this NIDS include:

  * **AI-Powered Network Traffic Classification:** An AI/ML model automatically categorizes network traffic from uploaded CSV files based on learned behavior and patterns.
  * **User-Friendly Web Interface (Streamlit):** An intuitive web application for easy CSV file upload, classification, result viewing, and downloading of classified data.
  * **Enhanced Threat Detection & Anomaly Identification:** An AI-driven security mechanism to identify suspicious or malicious activity, significantly improving overall threat detection.
  * **Optimized False Positive & False Negative Rates:** Aims to minimize incorrect alerts and, critically, missed attacks through rigorous model training and evaluation.
  * **Scalable Performance for Large Datasets:** Ensures the AI model can efficiently process and classify large volumes of network traffic data from CSV files.

-----

## 📊 Dataset

This project utilizes a comprehensive network traffic dataset obtained from Kaggle: [https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset).

The model was trained on a processed version of this dataset, derived from multiple Parquet files. This combined and preprocessed data (`final_netshield_cleaned_scaled_dataset.parquet`) is included in the `processed_data/` directory of this repository and is managed by Git LFS. For **demonstration and classification via the Streamlit application**, users are expected to upload network traffic data in **CSV format**, such as the provided `sample_http.csv`.

-----

## 🛠️ Methodology

### Data Preprocessing

The raw network data undergoes several preprocessing steps to prepare it for machine learning:

  * **Feature Extraction:** Deriving meaningful numerical features from raw network traffic data.
  * **Categorical Encoding:** Converting categorical features (e.g., protocol types, flags) into numerical representations using techniques like Label Encoding or One-Hot Encoding.
  * **Feature Scaling:** Normalizing numerical features to a standard range.
  * **Column Selection:** Identifying and selecting the most relevant features while excluding irrelevant or redundant ones.

### Class Imbalance Handling (SMOTE)

Network intrusion datasets are inherently imbalanced. To address this, the **Synthetic Minority Over-sampling Technique (SMOTE)** is employed on the training data. A custom sampling strategy ensures effective learning from under-represented attack patterns.

### Model Training (Random Forest)

A **Random Forest Classifier** was chosen for its robustness and performance. The model is trained on the SMOTE-resampled data to effectively learn the characteristics of both normal and various malicious traffic types.

### Evaluation

The model's performance is rigorously evaluated on an original, untouched test set. Key metrics include:

  * **Accuracy:** Overall correctness of predictions.
  * **Precision, Recall, F1-Score:** Detailed metrics for each class.
  * **Confusion Matrix:** Provides a breakdown of true positives, true negatives, false positives, and false negatives.
  * **False Negative Analysis:** A focused analysis on missed attack detections, critical for an IDS.

-----

## 📈 Key Results

The implementation of SMOTE with a custom sampling strategy significantly improved the model's ability to detect minority class attacks, leading to:

  * **High Overall Classification Performance:** Robust capabilities across various traffic types.
  * **Excellent Binary Classification:** Strong ability to distinguish between "Malicious" and "Not Malicious" traffic.
  * **Significant False Negative Reduction:** Minimized missed attack detections for critical attack types.
  * **Strong Multi-class Performance:** High classification capability across all defined attack categories.

-----

## 📂 File Structure

```
Network-Intrusion-Detection-System/
├── .streamlit/
│   └── config.toml
├── models/
│   └── rf_model_smote.pkl
│   └── scaler.joblib
│   └── label_encoder.joblib
├── processed_data/
│   └── final_netshield_cleaned_scaled_dataset.parquet
├── utils/
│   └── nids_core.py
├── .gitattributes
├── app.py
├── requirements.txt
├── sample_http.csv
├── netshield2.ipynb
├── train.py
├── nids.pptx
├── Network-Intrusion-Detection-System.pdf
└── README.md
```
📄 Documentation
📘 Project Report: Network-Intrusion-Detection-System.pdf
📑 Presentation Slides: nids.pptx
-----

## ⚙️ Setup and Installation

To get this project up and running on your local machine, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/SulfaSaji/net-sentinel.git
    cd net-sentinel
    ```

2.  **Install Git LFS:**
    If you don't have Git LFS installed, download and install it from [git-lfs.com](https://git-lfs.com). After installation, run:

    ```bash
    git lfs install
    ```

    This ensures that the large model file (`rf_model_smote.pkl`) and the processed dataset (`final_netshield_cleaned_scaled_dataset.parquet`) are correctly downloaded.

3.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    All required Python libraries are listed in `requirements.txt`. Make sure `streamlit` is included in this file.

    ```bash
    pip install -r requirements.txt
    ```

-----

## 🚀 Usage

### Running the Streamlit Application

The `app.py` is a Streamlit application providing a web interface for classification. A pre-trained model and its associated preprocessor artifacts are included in the `models/` directory, so you don't need to train the model yourself to run the application.

1.  **Prepare your input file:** You can use the provided `sample_http.csv` or convert one of the dataset's Parquet files into CSV format, as the Streamlit app expects CSV input.

2.  **Start the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

3.  **Open in Browser:** A new tab will automatically open in your web browser, directing you to the Streamlit application (usually `http://localhost:8501`).

4.  **Upload File:** Use the file uploader widget to select your prepared CSV file.

5.  **View Classification Results:** The application will process the file, display results, and show classification accuracy.

6.  **Download Classified File:** An option to download the processed file with classification labels will be available.

### Training the Model (For Developers)

The `train.py` script orchestrates the entire model training and evaluation process. This script is primarily for developers who wish to:

  * Retrain the model with new or updated datasets.
  * Understand or modify the full data preprocessing, imbalance handling, and training pipeline.
  * Experiment with different model architectures or hyperparameters.

**Note:** The detailed step-by-step development and exploration of the model, including initial data analysis and various iterations, are documented in the `netshield2.ipynb` Jupyter Notebook. The `train.py` script provides a streamlined, non-interactive version of the final training process.

To run the training process:

```bash
python train.py
```

This script will:

  * Load the processed dataset (`final_netshield_cleaned_scaled_dataset.parquet`).
  * Perform all preprocessing steps (fitting preprocessors on training data).
  * Apply SMOTE to the training data.
  * Train the Random Forest model.
  * Evaluate the model on a separate test set and print detailed reports.
  * Save the trained model (`models/rf_model_smote.pkl`) and fitted preprocessors (`scaler.joblib`, `label_encoder.joblib`).

-----

## 🔮 Future Work

  * **Robust Input Handling:** Enhance `app.py` to robustly handle various CSV file structures and provide more user-friendly error messages.
  * **Automated Retraining Pipeline:** Implement an automated pipeline for periodic model retraining to adapt to evolving threat landscapes.
  * **Enhanced Visualization:** Improve the Streamlit dashboard with more interactive plots and visualizations of classification results.
  * **Further Model Exploration:** Investigate other machine learning algorithms (e.g., Gradient Boosting, Neural Networks).
  * **Feature Engineering Enhancement:** Explore more advanced feature engineering techniques for specific attack types or encrypted traffic analysis.

-----

## 📄 License

This project is developed with **All Rights Reserved** by the authors and is not currently distributed under an open-source license.

-----

## 🙌 Acknowledgements

* **Dataset:** The dataset used in this project was obtained from Kaggle.
* **Libraries:** This project makes use of various open-source Python libraries including `pandas`, `scikit-learn`, `numpy`, `imbalanced-learn`, and `streamlit`.
* **Presentation:** You can view or download the project presentation slides [here](nids.pptx).
