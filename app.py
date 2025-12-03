import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler # Ensure these are imported

# Configure Streamlit page
st.set_page_config(layout='wide', page_title="AQI Predictor and Analyzer")

# --- 1. Load Pre-trained Assets (if available) ---
# Pretrained models expected filenames
BASE_PATH = '/content/drive/MyDrive/programing_data_analysis/'
RF_MODEL_FILE = BASE_PATH + 'random_forest_model.joblib'
KNN_MODEL_FILE = BASE_PATH + 'knn_model.joblib'
SCALER_FILE = BASE_PATH + 'scaler.joblib'
CITY_LE_FILE = BASE_PATH + 'city_label_encoder.joblib'
AQI_LE_FILE = BASE_PATH + 'aqi_bucket_label_encoder.joblib'
FEATURE_COLS_FILE = BASE_PATH + 'feature_columns.joblib'
DT_MODEL_FILE = BASE_PATH + 'decision_tree_model.joblib'

# Helper to safe-load joblib assets (returns None if not found)
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

rf_classifier = safe_load(RF_MODEL_FILE)
knn_classifier = safe_load(KNN_MODEL_FILE)
scaler = safe_load(SCALER_FILE)
city_label_encoder = safe_load(CITY_LE_FILE)
aqi_bucket_label_encoder = safe_load(AQI_LE_FILE)
X_cols = safe_load(FEATURE_COLS_FILE)
dt_classifier = safe_load(DT_MODEL_FILE)  # may be None; we'll train later if needed

# --- Load and Preprocess Data (for historical data, unique city list, and consistency) ---
df_display = pd.read_csv('all_cities_combined.csv')

# Re-apply preprocessing steps (based on your notebook)
df_display.drop_duplicates(inplace=True)

# Clean string columns
object_cols_to_clean = ['City', 'AQI_Bucket']
for col in object_cols_to_clean:
    if col in df_display.columns and df_display[col].dtype == 'object':
        df_display[col] = df_display[col].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)

# Impute numerical columns with median where present
numerical_cols_for_imputation = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
for col in numerical_cols_for_imputation:
    if col in df_display.columns:
        median_val = df_display[col].median()
        df_display[col] = df_display[col].fillna(median_val)

# Impute AQI_Bucket with mode
if 'AQI_Bucket' in df_display.columns:
    mode_val_aqi = df_display['AQI_Bucket'].mode()[0]
    df_display['AQI_Bucket'] = df_display['AQI_Bucket'].fillna(mode_val_aqi)

# Date conversion and temporal features
df_display['Date'] = pd.to_datetime(df_display['Date'], format='%d/%m/%Y', errors='coerce')
df_display.dropna(subset=['Date'], inplace=True)
df_display['Year'] = df_display['Date'].dt.year
df_display['Month'] = df_display['Date'].dt.month
df_display['Day'] = df_display['Date'].dt.day
df_display['Day_of_Week'] = df_display['Date'].dt.dayofweek

# Ensure encoders exist and are fitted to current unique values
if city_label_encoder is None:
    city_label_encoder = LabelEncoder()
city_label_encoder.fit(df_display['City'].unique())

if aqi_bucket_label_encoder is None:
    aqi_bucket_label_encoder = LabelEncoder()
aqi_bucket_label_encoder.fit(df_display['AQI_Bucket'].unique())

# Ensure feature columns exist (X_cols); if not, attempt to construct from expected names
if X_cols is None:
    # Typical feature order used in your app: pollutant columns + City_encoded + Year,Month,Day,Day_of_Week
    pollutant_cols_for_X = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
    X_cols = pollutant_cols_for_X + ['City_encoded','Year','Month','Day','Day_of_Week']

# --- Prepare X and y for potential training/evaluation of the Decision Tree ---
# Encode AQI_Bucket and City into numeric labels for modelling
df_display['AQI_Bucket_encoded'] = aqi_bucket_label_encoder.transform(df_display['AQI_Bucket'])
df_display['City_encoded'] = city_label_encoder.transform(df_display['City'])

# Ensure all X_cols are present in df_display; if some are missing, raise helpful error
missing_cols = [c for c in X_cols if c not in df_display.columns]
if missing_cols:
    st.error(f"The following expected feature columns are missing from your dataset: {missing_cols}. Please check 'feature_columns.joblib' or your CSV.")
    st.stop()

X_app = df_display[X_cols].copy()
y_app = df_display['AQI_Bucket_encoded'].copy()

# Numerical pollutant column names for scaling (must be subset of X_cols)
numerical_pollutant_cols_app = [c for c in ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene'] if c in X_app.columns]

# If scaler is missing, create and fit a simple scaler (MinMax or Standard); here we try StandardScaler
if scaler is None:
    scaler = StandardScaler()
    # fit scaler on numerical pollutant columns
    scaler.fit(X_app[numerical_pollutant_cols_app])

# Split for training/testing (we'll use this to train decision tree and evaluate all models)
X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(X_app, y_app, test_size=0.2, random_state=42, stratify=y_app) # Use 20% for consistency

# Scale numerical pollutant columns on train/test copies
X_train_scaled_app = X_train_app.copy()
X_test_scaled_app = X_test_app.copy()

if numerical_pollutant_cols_app:
    X_train_scaled_app[numerical_pollutant_cols_app] = scaler.transform(X_train_app[numerical_pollutant_cols_app])
    X_test_scaled_app[numerical_pollutant_cols_app] = scaler.transform(X_test_app[numerical_pollutant_cols_app])

# --- Decision Tree: load or train if not available ---
if dt_classifier is None:
    # Train a Decision Tree classifier on the training split
    dt_classifier = DecisionTreeClassifier(max_depth=None, random_state=42)
    try:
        dt_classifier.fit(X_train_scaled_app, y_train_app)
        # attempt to save the trained decision tree for reuse
        try:
            joblib.dump(dt_classifier, DT_MODEL_FILE)
        except Exception:
            # Not critical; just continue if saving fails
            pass
    except Exception as e:
        st.error(f"Failed to train Decision Tree classifier: {e}")
        st.stop()

# --- Prediction Function (now includes Decision Tree) ---
def predict_aqi_bucket(pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene, city, date_input):
    city = str(city).lower()
    try:
        city_encoded_val = city_label_encoder.transform([city])[0]
    except ValueError:
        return ("Error: City not recognized. Please select from the list.",
                "Error: City not recognized. Please select from the list.",
                "Error: City not recognized. Please select from the list.")
    # parse date
    input_date = pd.to_datetime(date_input)
    year = input_date.year
    month = input_date.month
    day = input_date.day
    day_of_week = input_date.dayofweek

    input_data_dict = {
        'PM2.5': [pm25], 'PM10': [pm10], 'NO': [no], 'NO2': [no2], 'NOx': [nox],
        'NH3': [nh3], 'CO': [co], 'SO2': [so2], 'O3': [o3], 'Benzene': [benzene],
        'Toluene': [toluene], 'Xylene': [xylene], 'City_encoded': [city_encoded_val],
        'Year': [year], 'Month': [month], 'Day': [day], 'Day_of_Week': [day_of_week]
    }
    input_data = pd.DataFrame(input_data_dict)[X_cols]

    # Scale numerical pollutant columns in the input
    if numerical_pollutant_cols_app and scaler is not None:
        input_data[numerical_pollutant_cols_app] = scaler.transform(input_data[numerical_pollutant_cols_app])

    # Random Forest prediction (if available)
    if rf_classifier is not None:
        rf_prediction_encoded = rf_classifier.predict(input_data)
        rf_prediction_label = aqi_bucket_label_encoder.inverse_transform(rf_prediction_encoded)[0]
    else:
        rf_prediction_label = "Random Forest not available"

    # KNN prediction (if available)
    if knn_classifier is not None:
        knn_prediction_encoded = knn_classifier.predict(input_data)
        knn_prediction_label = aqi_bucket_label_encoder.inverse_transform(knn_prediction_encoded)[0]
    else:
        knn_prediction_label = "KNN not available"

    # Decision Tree prediction (should be available)
    if dt_classifier is not None:
        dt_prediction_encoded = dt_classifier.predict(input_data)
        dt_prediction_label = aqi_bucket_label_encoder.inverse_transform(dt_prediction_encoded)[0]
    else:
        dt_prediction_label = "Decision Tree not available"

    return rf_prediction_label, knn_prediction_label, dt_prediction_label

# --- Multipage Navigation ---
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Data Overview', 'Exploratory Data Analysis', 'Modelling and Prediction'])

if page == 'Data Overview':
    st.title('Data Overview')
    st.write("### First 5 Rows of the Dataset")
    st.dataframe(df_display.head())

    st.write("### Dataset Shape")
    st.write(f"Number of Rows: {df_display.shape[0]}, Number of Columns: {df_display.shape[1]}")

    st.write("### Dataset Information")
    info_buffer = io.StringIO()
    df_display.info(verbose=True, buf=info_buffer)
    st.text(info_buffer.getvalue())

    st.write("### Descriptive Statistics for Numerical Columns")
    st.dataframe(df_display.describe())

    st.write("### Missing Values Table")
    def missing_values_table(df_local):
        mis_val = df_local.isnull().sum()
        mis_val_percent = 100 * mis_val / len(df_local)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
        mis_val_table = mis_val_table.sort_values('% of Total Values', ascending=False)
        return mis_val_table
    st.dataframe(missing_values_table(df_display))

    st.write("### Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_display.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Missing Values Heatmap')
    st.pyplot(fig)

elif page == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis')

    st.header('Annual Trends of Pollutant Levels')
    pollutant_cols_eda = [c for c in ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'] if c in df_display.columns]
    annual_mean_pollutants = df_display.groupby('Year')[pollutant_cols_eda].mean()

    # Create line plots (grid)
    fig_annual_trends, axes = plt.subplots(4, 3, figsize=(18, 15))
    axes = axes.flatten()
    for i, col in enumerate(pollutant_cols_eda):
        if i < len(axes): # Ensure we don't go out of bounds for subplots
            sns.lineplot(x=annual_mean_pollutants.index, y=col, data=annual_mean_pollutants, ax=axes[i])
            axes[i].set_title(f'Annual Mean of {col}')
            axes[i].set_xlabel('Year')
            axes[i].set_ylabel(f'Mean {col} Level')
    # hide any unused subplots
    for j in range(len(pollutant_cols_eda), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig_annual_trends)

    st.header('Mean AQI Across Cities')
    mean_aqi_per_city = df_display.groupby('City')['AQI'].mean()
    sorted_mean_aqi = mean_aqi_per_city.sort_values(ascending=False)

    fig_city_aqi, ax_city_aqi = plt.subplots(figsize=(15, 8))
    sns.barplot(x=sorted_mean_aqi.index, y=sorted_mean_aqi.values, ax=ax_city_aqi)
    ax_city_aqi.set_title('Mean AQI Across Cities', fontsize=16)
    ax_city_aqi.set_xlabel('City', fontsize=12)
    ax_city_aqi.set_ylabel('Mean AQI', fontsize=12)
    ax_city_aqi.set_xticks(range(len(sorted_mean_aqi.index)))
    ax_city_aqi.set_xticklabels(sorted_mean_aqi.index, rotation=90)
    plt.tight_layout()
    st.pyplot(fig_city_aqi)

    st.header('Pollutant Distributions by AQI Bucket')
    key_pollutants_eda = [p for p in ['PM2.5','PM10','NO2','CO','O3'] if p in df_display.columns]
    aqi_bucket_order = ['good', 'satisfactory', 'moderate', 'poor', 'very poor', 'severe']

    fig_aqi_dist, axes_aqi_dist = plt.subplots(2, 3, figsize=(18, 12))
    axes_aqi_dist = axes_aqi_dist.flatten()
    for i, col in enumerate(key_pollutants_eda):
        if i < len(axes_aqi_dist):
            sns.boxplot(x='AQI_Bucket', y=col, data=df_display, order=[b for b in aqi_bucket_order if b in df_display['AQI_Bucket'].unique()], ax=axes_aqi_dist[i], hue='AQI_Bucket', dodge=False, legend=False)
            axes_aqi_dist[i].set_title(f'Distribution of {col} by AQI Bucket')
            axes_aqi_dist[i].set_xlabel('AQI Bucket')
            axes_aqi_dist[i].set_ylabel(f'{col} Concentration')
            axes_aqi_dist[i].tick_params(axis='x', rotation=45)
    for j in range(len(key_pollutants_eda), len(axes_aqi_dist)):
        axes_aqi_dist[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig_aqi_dist)

    st.header('Correlation Matrix of Pollutants')
    numeric_df_corr = df_display.select_dtypes(include=['number'])
    corr_matrix = numeric_df_corr.corr(method='pearson')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", mask=mask, cmap='coolwarm', linewidths=0.5, cbar=True, ax=ax_corr)
    ax_corr.set_title('Correlation Matrix of Pollutants')
    st.pyplot(fig_corr)

    st.header('Skewness Analysis in Features')
    skew = df_display.select_dtypes(include=['number']).skew().sort_values(ascending=False)
    skew_df = pd.DataFrame(skew, columns=['Skewness'])

    fig_skew, ax_skew = plt.subplots(figsize=(10, 6))
    sns.barplot(x=skew_df.index, y=skew_df['Skewness'], edgecolor='black', linewidth=0.5, ax=ax_skew)
    ax_skew.set_xticks(range(len(skew_df.index)))
    ax_skew.set_xticklabels(skew_df.index, rotation=75)
    ax_skew.set_xlabel('Features')
    ax_skew.set_ylabel('Skewness')
    ax_skew.set_title('Skewness Analysis in Features')
    plt.tight_layout()
    st.pyplot(fig_skew)

elif page == 'Modelling and Prediction':
    st.title('AQI Prediction')
    st.write("Enter pollutant levels and city details to predict the AQI Bucket.")

    city_options = sorted(df_display['City'].unique())
    selected_city = st.selectbox('Select City', city_options)

    today = datetime.now().date()
    selected_date = st.date_input('Date', today)

    # Input sliders
    col1, col2, col3 = st.columns(3)
    with col1:
        pm25 = st.slider('PM2.5', 0.0, 500.0, float(df_display['PM2.5'].median() if 'PM2.5' in df_display.columns else 0.0))
        pm10 = st.slider('PM10', 0.0, 500.0, float(df_display['PM10'].median() if 'PM10' in df_display.columns else 0.0))
        no = st.slider('NO', 0.0, 200.0, float(df_display['NO'].median() if 'NO' in df_display.columns else 0.0))
        no2 = st.slider('NO2', 0.0, 200.0, float(df_display['NO2'].median() if 'NO2' in df_display.columns else 0.0))
    with col2:
        nox = st.slider('NOx', 0.0, 200.0, float(df_display['NOx'].median() if 'NOx' in df_display.columns else 0.0))
        nh3 = st.slider('NH3', 0.0, 200.0, float(df_display['NH3'].median() if 'NH3' in df_display.columns else 0.0))
        co = st.slider('CO', 0.0, 50.0, float(df_display['CO'].median() if 'CO' in df_display.columns else 0.0))
        so2 = st.slider('SO2', 0.0, 100.0, float(df_display['SO2'].median() if 'SO2' in df_display.columns else 0.0))
    with col3:
        o3 = st.slider('O3', 0.0, 200.0, float(df_display['O3'].median() if 'O3' in df_display.columns else 0.0))
        benzene = st.slider('Benzene', 0.0, 50.0, float(df_display['Benzene'].median() if 'Benzene' in df_display.columns else 0.0))
        toluene = st.slider('Toluene', 0.0, 100.0, float(df_display['Toluene'].median() if 'Toluene' in df_display.columns else 0.0))
        xylene = st.slider('Xylene', 0.0, 50.0, float(df_display['Xylene'].median() if 'Xylene' in df_display.columns else 0.0))

    if st.button('Predict AQI Bucket'):
        rf_pred, knn_pred, dt_pred = predict_aqi_bucket(pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene, selected_city, selected_date)
        st.subheader('Prediction Results:')
        # RF
        if isinstance(rf_pred, str) and rf_pred.startswith("Error"):
            st.error(rf_pred)
        else:
            st.success(f"Random Forest Prediction: **{rf_pred.upper()}**") if rf_classifier is not None else st.warning("Random Forest model not available")
        # KNN
        if isinstance(knn_pred, str) and knn_pred.startswith("Error"):
            st.error(knn_pred)
        else:
            st.info(f"KNN Prediction: **{knn_pred.upper()}**") if knn_classifier is not None else st.warning("KNN model not available")
        # Decision Tree
        if isinstance(dt_pred, str) and dt_pred.startswith("Error"):
            st.error(dt_pred)
        else:
            st.success(f"Decision Tree Prediction: **{dt_pred.upper()}**")

    # --- Model Performance Section (Random Forest, KNN, Decision Tree) ---
    st.subheader("Model Performance")

    # Make predictions for each classifier on the scaled test data
    y_pred_rf_app = rf_classifier.predict(X_test_scaled_app) if rf_classifier is not None else None
    y_pred_knn_app = knn_classifier.predict(X_test_scaled_app) if knn_classifier is not None else None
    y_pred_dt_app = dt_classifier.predict(X_test_scaled_app) if dt_classifier is not None else None

    # Calculate metrics for all models
    metrics_data = {}

    def extract_metrics_for_streamlit(y_true, y_pred, model_name, metrics_dict):
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics_dict[model_name] = {
            'Accuracy': report['accuracy'],
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score']
        }

    if y_pred_rf_app is not None:
        extract_metrics_for_streamlit(y_test_app, y_pred_rf_app, 'Random Forest', metrics_data)
    if y_pred_knn_app is not None:
        extract_metrics_for_streamlit(y_test_app, y_pred_knn_app, 'K-Nearest Neighbors', metrics_data)
    if y_pred_dt_app is not None:
        extract_metrics_for_streamlit(y_test_app, y_pred_dt_app, 'Decision Tree', metrics_data)

    classification_metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')

    st.write("### Model Accuracy Comparison")
    if not classification_metrics_df.empty:
        fig_accuracy, ax_accuracy = plt.subplots(figsize=(8, 6))
        sns.barplot(x=classification_metrics_df.index, y=classification_metrics_df['Accuracy'], palette='viridis', hue=classification_metrics_df.index, legend=False, ax=ax_accuracy)
        ax_accuracy.set_title('Comparison of Model Accuracy', fontsize=16)
        ax_accuracy.set_xlabel('Model', fontsize=12)
        ax_accuracy.set_ylabel('Accuracy', fontsize=12)
        ax_accuracy.set_ylim(0, 1)
        ax_accuracy.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig_accuracy)

        st.write("### Model Precision, Recall, and F1-Score Comparison")
        metrics_melted = classification_metrics_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
        metrics_melted = metrics_melted[metrics_melted['Metric'] != 'Accuracy']

        fig_other_metrics, ax_other_metrics = plt.subplots(figsize=(12, 7))
        sns.barplot(x='index', y='Score', hue='Metric', data=metrics_melted, palette='viridis', ax=ax_other_metrics)
        ax_other_metrics.set_title('Comparison of Model Precision, Recall, and F1-Score', fontsize=16)
        ax_other_metrics.set_xlabel('Model', fontsize=12)
        ax_other_metrics.set_ylabel('Score', fontsize=12)
        ax_other_metrics.set_ylim(0, 1)
        ax_other_metrics.grid(axis='y', linestyle='--', alpha=0.7)
        ax_other_metrics.legend(title='Metric')
        st.pyplot(fig_other_metrics)
    else:
        st.warning("No models were available to generate comparison metrics.")


    st.write("---")
    st.write("### Random Forest Classifier Performance (if available)")
    if rf_classifier is not None and y_pred_rf_app is not None:
        st.text("Classification Report (Random Forest):")
        st.code(classification_report(y_test_app, y_pred_rf_app, zero_division=0))
        fig_rf_cm, ax_rf_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test_app, y_pred_rf_app), annot=True, fmt='d', cmap='Blues', ax=ax_rf_cm, xticklabels=aqi_bucket_label_encoder.classes_, yticklabels=aqi_bucket_label_encoder.classes_)
        ax_rf_cm.set_title('Random Forest Confusion Matrix')
        ax_rf_cm.set_xlabel('Predicted Label')
        ax_rf_cm.set_ylabel('True Label')
        st.pyplot(fig_rf_cm)
    else:
        st.warning("Random Forest classifier not loaded. Skipping RF performance metrics.")

    st.write("---")
    st.write("### K-Nearest Neighbors Classifier Performance (if available)")
    if knn_classifier is not None and y_pred_knn_app is not None:
        st.text("Classification Report (KNN):")
        st.code(classification_report(y_test_app, y_pred_knn_app, zero_division=0))
        fig_knn_cm, ax_knn_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test_app, y_pred_knn_app), annot=True, fmt='d', cmap='Blues', ax=ax_knn_cm, xticklabels=aqi_bucket_label_encoder.classes_, yticklabels=aqi_bucket_label_encoder.classes_)
        ax_knn_cm.set_title('KNN Confusion Matrix')
        ax_knn_cm.set_xlabel('Predicted Label')
        ax_knn_cm.set_ylabel('True Label')
        st.pyplot(fig_knn_cm)
    else:
        st.warning("KNN classifier not loaded. Skipping KNN performance metrics.")

    st.write("---")
    st.write("### Decision Tree Classifier Performance")
    if dt_classifier is not None and y_pred_dt_app is not None:
        st.text("Classification Report (Decision Tree):")
        st.code(classification_report(y_test_app, y_pred_dt_app, zero_division=0))

        fig_dt_cm, ax_dt_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test_app, y_pred_dt_app), annot=True, fmt='d', cmap='Blues', ax=ax_dt_cm, xticklabels=aqi_bucket_label_encoder.classes_, yticklabels=aqi_bucket_label_encoder.classes_)
        ax_dt_cm.set_title('Decision Tree Confusion Matrix')
        ax_dt_cm.set_xlabel('Predicted Label')
        ax_dt_cm.set_ylabel('True Label')
        st.pyplot(fig_dt_cm)

        st.write("#### Visualizing the Decision Tree")
        try:
            plot_depth = 4
            fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
            # Filter X_cols to only include those that are actually features and in the correct order for the tree
            feature_names_for_tree = [col for col in X_cols if col in X_test_scaled_app.columns]
            plot_tree(dt_classifier, feature_names=feature_names_for_tree, class_names=[str(c) for c in aqi_bucket_label_encoder.classes_], filled=True, max_depth=plot_depth, fontsize=8, ax=ax_tree)
            ax_tree.set_title(f'Decision Tree (visualized up to depth={plot_depth})')
            st.pyplot(fig_tree)
        except Exception as e:
            st.warning(f"Could not plot decision tree due to: {e}")
    else:
        st.warning("Decision Tree classifier not loaded. Skipping DT performance metrics.")


# Sidebar credit
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by: Colab AI Agent")
