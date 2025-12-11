%%writefile app.py
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import math


# -------------------------
# Streamlit: AQI Predictor
# -------------------------

# Configure Streamlit
st.set_page_config(layout='wide', page_title="AQI Predictor and Analyzer")
st.title("AQI Predictor and Analyzer")

# -------------------------
# Configuration / Paths
# -------------------------
# *** CRITICAL FIX: Using a portable relative path 'assets/' ***
# This path assumes your models and preprocessors will be saved 
# in an 'assets' folder relative to where app.py is run.
BASE_PATH = 'assets/' 

if not BASE_PATH.endswith('/'):
    BASE_PATH += '/'

# Filenames for optional pre-trained assets
RF_MODEL_FILE = BASE_PATH + 'random_forest_model.joblib'
KNN_MODEL_FILE = BASE_PATH + 'knn_model.joblib'
SCALER_FILE = BASE_PATH + 'scaler.joblib'
CITY_LE_FILE = BASE_PATH + 'city_label_encoder.joblib'
AQI_LE_FILE = BASE_PATH + 'aqi_bucket_label_encoder.joblib'
FEATURE_COLS_FILE = BASE_PATH + 'feature_columns.joblib'
DT_MODEL_FILE = BASE_PATH + 'decision_tree_model.joblib'

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def safe_joblib_load(path):
    """Safely loads a joblib file, returning None if loading fails."""
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_data
def load_csv_from_path(path):
    """Safely loads a CSV file from a given path."""
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def ensure_dir(path):
    """Ensures the directory for a given path exists."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# Helper for Model Download (From your screenshot)
def get_joblib_download_link(model_object, filename, label="Download Model"):
    """
    Helper to convert a joblib model object to bytes and create a Streamlit download button.
    """
    buffer = io.BytesIO()
    joblib.dump(model_object, buffer)
    buffer.seek(0)
    
    # Create the download button
    return st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/octet-stream",
        key=f'download_{filename}'
    )


# -------------------------
# Upload or load dataset
# -------------------------
st.sidebar.header('Data & Model Options')
use_uploader = st.sidebar.checkbox('Upload CSV instead of using default path', value=True)

uploaded_file = None
if use_uploader:
    uploaded_file = st.sidebar.file_uploader('Upload all_cities_combined.csv', type=['csv'])

if uploaded_file is not None:
    try:
        df_display = pd.read_csv(uploaded_file)
        st.sidebar.success('Dataset uploaded successfully')
    except Exception as e:
        st.sidebar.error(f'Could not read uploaded CSV: {e}')
        st.stop()
else:
    # fallback to default path
    # NOTE: The CSV must be accessible at BASE_PATH + 'all_cities_combined.csv'
    df_display = load_csv_from_path(BASE_PATH + 'all_cities_combined.csv')
    if df_display is None:
        st.sidebar.warning(f'No uploaded file and default CSV not found at BASE_PATH: {BASE_PATH}. Use the uploader or place your CSV.')
        st.stop()
    else:
        st.sidebar.info(f'Loaded dataset from {BASE_PATH}all_cities_combined.csv')

# -------------------------
# Basic cleaning & preprocessing (defensive)
# -------------------------
# Drop exact duplicate rows
df_display = df_display.copy()
df_display.drop_duplicates(inplace=True)

# Standardize column names (strip)
df_display.columns = [c.strip() for c in df_display.columns]

# Required columns list (common expectation)
expected_numeric = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene','AQI']
required_cols = ['City','Date','AQI_Bucket'] + [c for c in expected_numeric if c in df_display.columns]

missing_required = [c for c in ['City','Date','AQI_Bucket'] if c not in df_display.columns]
if missing_required:
    st.error(f"Dataset is missing required columns: {missing_required}. Please provide them.")
    st.stop()

# Clean string columns
for col in ['City','AQI_Bucket']:
    if col in df_display.columns:
        df_display[col] = df_display[col].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)

# Convert Date column (attempt common formats)
possible_formats = ['%d/%m/%Y','%Y-%m-%d','%d-%m-%Y','%m/%d/%Y']
parsed_dates = None
for fmt in possible_formats:
    try:
        parsed_dates = pd.to_datetime(df_display['Date'], format=fmt, errors='coerce')
        if parsed_dates.notna().sum() > 0:
            df_display['Date'] = parsed_dates
            break
    except Exception:
        pass
# If still not parsed, try general parser
if df_display['Date'].dtype == object or df_display['Date'].isnull().any():
    df_display['Date'] = pd.to_datetime(df_display['Date'], errors='coerce')

if df_display['Date'].isnull().all():
    st.error('Could not parse any dates in the Date column. Ensure Date values are valid.')
    st.stop()

# Drop rows where Date is missing
df_display.dropna(subset=['Date'], inplace=True)

# Temporal features
df_display['Year'] = df_display['Date'].dt.year
df_display['Month'] = df_display['Date'].dt.month
df_display['Day'] = df_display['Date'].dt.day
df_display['Day_of_Week'] = df_display['Date'].dt.dayofweek

# Impute numeric columns with median (if they exist)
for col in expected_numeric:
    if col in df_display.columns:
        median_val = df_display[col].median()
        df_display[col] = df_display[col].fillna(median_val)

# Impute AQI_Bucket with mode
if 'AQI_Bucket' in df_display.columns:
    if df_display['AQI_Bucket'].isnull().any():
        df_display['AQI_Bucket'] = df_display['AQI_Bucket'].fillna(df_display['AQI_Bucket'].mode()[0])

# -------------------------
# Load or fit encoders and scaler
# -------------------------
city_label_encoder = safe_joblib_load(CITY_LE_FILE)
aqi_bucket_label_encoder = safe_joblib_load(AQI_LE_FILE)
scaler = safe_joblib_load(SCALER_FILE)
X_cols = safe_joblib_load(FEATURE_COLS_FILE)
rf_classifier = safe_joblib_load(RF_MODEL_FILE)
knn_classifier = safe_joblib_load(KNN_MODEL_FILE)
dt_classifier = safe_joblib_load(DT_MODEL_FILE)

# Fit encoders if missing
if city_label_encoder is None:
    city_label_encoder = LabelEncoder()
    city_label_encoder.fit(df_display['City'].unique())
    try:
        ensure_dir(CITY_LE_FILE)
        joblib.dump(city_label_encoder, CITY_LE_FILE)
    except Exception:
        pass

if aqi_bucket_label_encoder is None:
    aqi_bucket_label_encoder = LabelEncoder()
    aqi_bucket_label_encoder.fit(df_display['AQI_Bucket'].unique())
    try:
        ensure_dir(AQI_LE_FILE)
        joblib.dump(aqi_bucket_label_encoder, AQI_LE_FILE)
    except Exception:
        pass

# Construct X_cols if missing
if X_cols is None:
    pollutant_cols_for_X = [c for c in expected_numeric if c in df_display.columns]
    X_cols = pollutant_cols_for_X + ['City_encoded','Year','Month','Day','Day_of_Week']
    try:
        ensure_dir(FEATURE_COLS_FILE)
        joblib.dump(X_cols, FEATURE_COLS_FILE)
    except Exception:
        pass

# Encode numeric labels for modelling
try:
    df_display['AQI_Bucket_encoded'] = aqi_bucket_label_encoder.transform(df_display['AQI_Bucket'])
except Exception:
    # In case labels differ, refit encoder
    aqi_bucket_label_encoder.fit(df_display['AQI_Bucket'].unique())
    df_display['AQI_Bucket_encoded'] = aqi_bucket_label_encoder.transform(df_display['AQI_Bucket'])
    try:
        joblib.dump(aqi_bucket_label_encoder, AQI_LE_FILE)
    except Exception:
        pass

# City encoding
try:
    df_display['City_encoded'] = city_label_encoder.transform(df_display['City'])
except Exception:
    city_label_encoder.fit(df_display['City'].unique())
    df_display['City_encoded'] = city_label_encoder.transform(df_display['City'])
    try:
        joblib.dump(city_label_encoder, CITY_LE_FILE)
    except Exception:
        pass

# Ensure feature columns exist in dataframe
missing_cols = [c for c in X_cols if c not in df_display.columns]
if missing_cols:
    st.error(f"The following expected feature columns are missing from your dataset: {missing_cols}. Please check your CSV or feature configuration.")
    st.stop()

# Prepare X and y
X_app = df_display[X_cols].copy()
y_app = df_display['AQI_Bucket_encoded'].copy()

numerical_pollutant_cols_app = [c for c in expected_numeric if c in X_app.columns]

# Fit scaler if missing
if scaler is None:
    scaler = StandardScaler()
    scaler.fit(X_app[numerical_pollutant_cols_app])
    try:
        ensure_dir(SCALER_FILE)
        joblib.dump(scaler, SCALER_FILE)
    except Exception:
        pass

# Scale features copy for training/evaluation
X_scaled_app = X_app.copy()
if numerical_pollutant_cols_app:
    X_scaled_app[numerical_pollutant_cols_app] = scaler.transform(X_scaled_app[numerical_pollutant_cols_app])

# Train/test split
X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(X_scaled_app, y_app, test_size=0.2, random_state=42, stratify=y_app)

# -------------------------
# Train Decision Tree if missing (and allow optional training of other models)
# -------------------------
train_full_models = st.sidebar.checkbox('Train / retrain models now (this will train RF, KNN, DT)', value=False)

if dt_classifier is None or train_full_models:
    st.sidebar.info('Training Decision Tree (and optionally other models) ...')
    # Decision Tree
    dt_classifier = DecisionTreeClassifier(random_state=42)
    try:
        dt_classifier.fit(X_train_app, y_train_app)
        ensure_dir(DT_MODEL_FILE)
        joblib.dump(dt_classifier, DT_MODEL_FILE)
        st.sidebar.success('Decision Tree trained and saved.')
    except Exception as e:
        # The base path is relative now, so this error usually means file permissions 
        # or the environment doesn't allow file saving (e.g., read-only deployment).
        st.sidebar.error(f'Failed to train Decision Tree and save the model: {e}. Check if the directory "{BASE_PATH}" is writeable.')

if (rf_classifier is None or knn_classifier is None) and train_full_models:
    # Train Random Forest and KNN as optional
    try:
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train_app, y_train_app)
        ensure_dir(RF_MODEL_FILE)
        joblib.dump(rf_classifier, RF_MODEL_FILE)
        st.sidebar.success('Random Forest trained and saved.')
    except Exception as e:
        st.sidebar.warning(f'Random Forest training failed: {e}')

    try:
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(X_train_app, y_train_app)
        ensure_dir(KNN_MODEL_FILE)
        joblib.dump(knn_classifier, KNN_MODEL_FILE)
        st.sidebar.success('KNN trained and saved.')
    except Exception as e:
        st.sidebar.warning(f'KNN training failed: {e}')

# -------------------------
# Asset Download Section
# -------------------------
st.sidebar.markdown('---')
st.sidebar.subheader('Download Assets')

with st.sidebar.expander("Download Preprocessors"):
    if scaler:
        get_joblib_download_link(scaler, os.path.basename(SCALER_FILE), label="Download Scaler")
    else:
        st.caption("Scaler not loaded.")
        
    if city_label_encoder:
        get_joblib_download_link(city_label_encoder, os.path.basename(CITY_LE_FILE), label="Download City Encoder")
    else:
        st.caption("City Encoder not loaded.")
        
    if aqi_bucket_label_encoder:
        get_joblib_download_link(aqi_bucket_label_encoder, os.path.basename(AQI_LE_FILE), label="Download AQI Encoder")
    else:
        st.caption("AQI Encoder not loaded.")

with st.sidebar.expander("Download Trained Models"):
    if dt_classifier:
        get_joblib_download_link(dt_classifier, os.path.basename(DT_MODEL_FILE), label="Download Decision Tree")
    else:
        st.caption("Decision Tree not loaded.")
        
    if rf_classifier:
        get_joblib_download_link(rf_classifier, os.path.basename(RF_MODEL_FILE), label="Download Random Forest")
    else:
        st.caption("Random Forest not loaded.")
        
    if knn_classifier:
        get_joblib_download_link(knn_classifier, os.path.basename(KNN_MODEL_FILE), label="Download KNN")
    else:
        st.caption("KNN not loaded.")


# -------------------------
# Multipage style navigation
# -------------------------
page = st.sidebar.radio('Go to', ['Data Overview', 'Exploratory Data Analysis', 'Exploratory Data Analysis 2', 'Modelling and Prediction', 'Model Visualization'])

if page == 'Data Overview':
    st.header('Data Overview')
    st.write('First 5 rows:')
    st.dataframe(df_display.head())

    st.write('Dataset shape:')
    st.write(f'Rows: {df_display.shape[0]}, Columns: {df_display.shape[1]}')

    st.write('### Dataset Info')
    buf = io.StringIO()
    df_display.info(buf=buf)
    st.text(buf.getvalue())

    st.write('### Descriptive statistics')
    st.dataframe(df_display.describe())

    st.write('### Missing values table')
    mis_val = df_display.isnull().sum()
    mis_pct = 100 * mis_val / len(df_display)
    mv_table = pd.concat([mis_val, mis_pct], axis=1)
    mv_table.columns = ['Missing Values', '% of Total Values']
    mv_table = mv_table.sort_values('% of Total Values', ascending=False)
    st.dataframe(mv_table)

elif page == 'Exploratory Data Analysis':
    st.header('Exploratory Data Analysis 📊')

    if 'AQI' in df_display.columns:
        sorted_mean_aqi = df_display.groupby('City')['AQI'].mean().sort_values(ascending=False)
    else:
        sorted_mean_aqi = pd.Series()

    num_df = df_display.select_dtypes(include=['number']).copy()
    if num_df.shape[1] > 1:
        corr_matrix = num_df.corr()
    else:
        corr_matrix = pd.DataFrame()

    eda_sub_page = st.selectbox(
        'Select EDA View',
        ['Overview & Trends', 'Detailed Distributions & Correlations'],
        key='eda_sub_nav'
    )

    if eda_sub_page == 'Overview & Trends':
        st.subheader('Annual Pollutant Trends')
        pollutant_cols_eda = [c for c in expected_numeric if c in df_display.columns]
        if len(pollutant_cols_eda) > 0:
            annual = df_display.groupby('Year')[pollutant_cols_eda].mean()
            st.write('Annual Mean Pollutant Levels (Table)')
            st.dataframe(annual)

            st.write('Annual Trends (Line Plots)')
            n_plots = len(pollutant_cols_eda)
            n_rows = math.ceil(n_plots / 3)
            fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows))
            axes = np.array(axes).flatten()

            for i, col in enumerate(pollutant_cols_eda):
                ax = axes[i]
                sns.lineplot(x=annual.index, y=annual[col], ax=ax)
                ax.set_title(col)

            for j in range(len(pollutant_cols_eda), len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            st.pyplot(fig)


        # Mean AQI across cities (simple plot)
        if 'AQI' in df_display.columns and not sorted_mean_aqi.empty:
            st.subheader('Mean AQI Across Cities')
            fig2, ax2 = plt.subplots(figsize=(12,6))
            sns.barplot(x=sorted_mean_aqi.index, y=sorted_mean_aqi.values, ax=ax2, palette='coolwarm')
            ax2.set_xticklabels(sorted_mean_aqi.index, rotation=90)
            st.pyplot(fig2)

        # Correlation matrix
        st.subheader('Correlation Matrix')
        if not corr_matrix.empty:
            fig3, ax3 = plt.subplots(figsize=(12,10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', ax=ax3, cmap='viridis')
            st.pyplot(fig3)

        # Distribution of Categorical Features
        st.subheader('Distribution of Categorical Features')
        categorical_cols = df_display.select_dtypes(include='object').columns.tolist()

        if categorical_cols:
            n_cols = len(categorical_cols)
            fig_width = max(10, 5 * n_cols)
            fig_height = 6

            fig_cat, axes_cat = plt.subplots(1, n_cols, figsize=(fig_width, fig_height))

            if n_cols == 1:
                axes_cat = [axes_cat]

            for i, col in enumerate(categorical_cols):
                ax = axes_cat[i]
                sns.countplot(x=df_display[col], ax=ax, palette='viridis')
                ax.set_title(f'Distribution of {col}', fontsize=14)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            st.pyplot(fig_cat)

    elif eda_sub_page == 'Detailed Distributions & Correlations':
        st.subheader('Detailed Distributions of Key Pollutants')

        cols_hist = ['PM2.5','PM10','NO2','NOx','NH3', 'CO','SO2','O3','Benzene','Toluene','Xylene']
        cols_hist = [c for c in cols_hist if c in df_display.columns]

        if cols_hist:
            st.write('Individual Histograms of Key Pollutants')

            n_plots = len(cols_hist)
            n_rows = math.ceil(n_plots / 4)
            fig_indiv_hist, axes_indiv_hist = plt.subplots(n_rows, 4, figsize=(18, 5 * n_rows))
            axes_indiv_hist = np.array(axes_indiv_hist).flatten()

            for i, col in enumerate(cols_hist):
                ax = axes_indiv_hist[i]
                ax.hist(df_display[col].dropna(), bins=20, edgecolor='black', color=sns.color_palette('pastel')[i % 10])
                ax.set_title(f"Histogram of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.grid(True, alpha=0.3)

            for j in range(len(cols_hist), len(axes_indiv_hist)):
                axes_indiv_hist[j].axis("off")

            plt.tight_layout()
            st.pyplot(fig_indiv_hist)

        st.subheader('Feature Skewness Analysis')
        if not num_df.empty:
            skew = num_df.skew().sort_values(ascending=False)
            skew_df = pd.DataFrame(skew, columns=['Skewness'])

            fig_skew, ax_skew = plt.subplots(figsize=(10, 6))
            sns.barplot(x=skew_df.index, y=skew_df['Skewness'],
                        edgecolor='black', linewidth=0.5, palette='viridis_r', ax=ax_skew)

            ax_skew.set_xticks(range(len(skew_df.index)))
            ax_skew.set_xticklabels(skew_df.index, rotation=75, ha='right')
            ax_skew.set_xlabel('Features')
            ax_skew.set_ylabel('Skewness')
            ax_skew.set_title('Skewness Analysis in Features')
            plt.tight_layout()
            st.pyplot(fig_skew)

        st.subheader('Grid of All Numerical Features Distributions (EDA 1)')
        numerical_cols = df_display.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        exclude_for_hist_eda1 = ['City_encoded', 'AQI_Bucket_encoded', 'Year']
        numerical_cols_for_hist_eda1 = [col for col in numerical_cols if col not in exclude_for_hist_eda1 and col not in ['Date', 'Day_of_Week', 'Month', 'Day']]

        if numerical_cols_for_hist_eda1:
            n_plots = len(numerical_cols_for_hist_eda1)
            n_rows = math.ceil(n_plots / 4)

            fig_grid_hist, axes_grid_hist = plt.subplots(n_rows, 4, figsize=(18, 4 * n_rows))
            axes_grid_hist = np.array(axes_grid_hist).flatten()

            for i, col in enumerate(numerical_cols_for_hist_eda1):
                ax = axes_grid_hist[i]
                sns.histplot(df_display[col], kde=True, bins=30, ax=ax, color=sns.color_palette('magma')[i % 6])
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')

            for j in range(n_plots, len(axes_grid_hist)):
                axes_grid_hist[j].axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.suptitle('Histograms of Numerical Features (EDA 1)', y=1.0, fontsize=18)
            st.pyplot(fig_grid_hist)

        st.subheader('Distribution of Key Pollutants by AQI Bucket')
        key_pollutants = [p for p in ['PM2.5', 'PM10', 'NO2', 'CO', 'O3'] if p in df_display.columns]
        aqi_order = ['good', 'satisfactory', 'moderate', 'poor', 'very poor', 'severe']

        if key_pollutants:
            n_plots = len(key_pollutants)
            n_rows = math.ceil(n_plots / 3)
            fig_box, axes_box = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
            axes_box = np.array(axes_box).flatten()

            for i, col in enumerate(key_pollutants):
                ax = axes_box[i]
                sns.boxplot(x='AQI_Bucket', y=col, data=df_display,
                            order=[b for b in aqi_order if b in df_display['AQI_Bucket'].unique()],
                            ax=ax, palette='Spectral')
                ax.set_title(f'Distribution of {col} by AQI Bucket')
                ax.set_xlabel('AQI Bucket')
                ax.set_ylabel(f'{col} Concentration')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            for j in range(len(key_pollutants), len(axes_box)):
                axes_box[j].axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.suptitle('Distribution of Key Pollutants by AQI Bucket', y=1.0, fontsize=18)
            st.pyplot(fig_box)


elif page == 'Exploratory Data Analysis 2':
    st.header('Exploratory Data Analysis 2: Custom Trends & Correlation 📈')

    if 'AQI' in df_display.columns:
        sorted_mean_aqi = df_display.groupby('City')['AQI'].mean().sort_values(ascending=False)
    else:
        sorted_mean_aqi = pd.Series()

    num_df = df_display.select_dtypes(include=['number']).copy()
    if num_df.shape[1] > 1:
        corr_matrix = num_df.corr()
    else:
        corr_matrix = pd.DataFrame()


    st.subheader('1. Annual AQI Trend for Top 5 Cities')

    if 'AQI' in df_display.columns and not sorted_mean_aqi.empty:
        top_5_cities = sorted_mean_aqi.head(5).index.tolist()

        df_top_cities = df_display[df_display['City'].isin(top_5_cities)]

        annual_aqi_top_cities = df_top_cities.groupby(['Year', 'City'])['AQI'].mean().reset_index()

        fig_trend, ax_trend = plt.subplots(figsize=(12, 7))
        sns.lineplot(x='Year', y='AQI', hue='City', data=annual_aqi_top_cities, marker='o', ax=ax_trend)
        ax_trend.set_title('Annual AQI Trend for Top 5 Cities', fontsize=16)
        ax_trend.set_xlabel('Year', fontsize=12)
        ax_trend.set_ylabel('Mean AQI', fontsize=12)
        ax_trend.set_xticks(annual_aqi_top_cities['Year'].unique())
        ax_trend.grid(True, linestyle='--', alpha=0.6)
        ax_trend.legend(title='City')
        plt.tight_layout()
        st.pyplot(fig_trend)
    else:
        st.warning('AQI data is required to show Top 5 Cities trend.')


    st.subheader('2. Top 10 Cities by Mean AQI')

    if 'AQI' in df_display.columns and not sorted_mean_aqi.empty:
        fig_top_cities, ax_top_cities = plt.subplots(figsize=(12, 6))
        top_10 = sorted_mean_aqi.head(10)

        sns.barplot(x=top_10.index, y=top_10.values, hue=top_10.index, palette='viridis', legend=False, ax=ax_top_cities)
        ax_top_cities.set_title('Top 10 Cities by Mean AQI')
        ax_top_cities.set_xlabel('City')
        ax_top_cities.set_ylabel('Mean AQI')
        ax_top_cities.set_xticklabels(top_10.index, rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_top_cities)
    else:
        st.warning('AQI data is required to calculate and show the Top 10 Cities by Mean AQI.')


    st.subheader('3. Histograms of Numerical Features')

    numerical_cols = df_display.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
    exclude_for_hist = ['City_encoded', 'AQI_Bucket_encoded', 'Year', 'Month', 'Day', 'Day_of_Week']
    numerical_cols_for_hist = [col for col in numerical_cols if col not in exclude_for_hist and col not in ['Date']]

    if numerical_cols_for_hist:
        n_plots = len(numerical_cols_for_hist)
        grid_size = math.ceil(math.sqrt(n_plots))

        fig_hist_eda2, axes_hist_eda2 = plt.subplots(grid_size, grid_size, figsize=(18, 5 * grid_size))
        axes_hist_eda2 = np.array(axes_hist_eda2).flatten()

        for i, col in enumerate(numerical_cols_for_hist):
            ax = axes_hist_eda2[i]
            sns.histplot(df_display[col], kde=True, bins=30, ax=ax, color=sns.color_palette('magma')[i % 6])
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')

        for j in range(n_plots, len(axes_hist_eda2)):
            axes_hist_eda2[j].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.suptitle('Histograms of Numerical Features (EDA 2)', y=1.0, fontsize=18)
        st.pyplot(fig_hist_eda2)
    else:
        st.warning('No suitable numerical columns found for histogram plotting.')


    st.subheader('4. Distribution of AQI Buckets (Pie Chart) 🥧')
    if 'AQI_Bucket' in df_display.columns:
        aqi_bucket_counts = df_display['AQI_Bucket'].value_counts()
        aqi_bucket_percentages = (aqi_bucket_counts / len(df_display)) * 100

        fig_pie, ax_pie = plt.subplots(figsize=(10, 8))
        ax_pie.pie(aqi_bucket_percentages, labels=aqi_bucket_percentages.index,
                      autopct='%1.1f%%', startangle=90,
                      colors=sns.color_palette('viridis', len(aqi_bucket_percentages)),
                      wedgeprops={'edgecolor': 'black', 'linewidth': 1})
        ax_pie.set_title('Distribution of AQI Buckets')
        ax_pie.axis('equal') 
        st.pyplot(fig_pie)
        st.dataframe(aqi_bucket_percentages.to_frame(name='Percentage (%)'))
        st.write('Pie chart showing the distribution of AQI Buckets has been generated.')

    else:
        st.warning('The AQI_Bucket column is required to generate the Pie Chart.')


    st.subheader('5. Highly Correlated Pollutant Pairs Analysis')

    if not corr_matrix.empty:
        highly_correlated_pairs = []

        pollutant_cols = [c for c in expected_numeric if c in corr_matrix.columns and c != 'AQI']
        corr_pollutants = corr_matrix.loc[pollutant_cols, pollutant_cols]

        for i in range(len(corr_pollutants.columns)):
            for j in range(i + 1, len(corr_pollutants.columns)):
                corr_val = corr_pollutants.iloc[i, j]
                if abs(corr_val) > 0.7: 
                    col1 = corr_pollutants.columns[i]
                    col2 = corr_pollutants.columns[j]
                    highly_correlated_pairs.append((col1, col2, corr_val))

        if highly_correlated_pairs:
            st.write("Highly correlated pollutant pairs (absolute correlation > 0.7):")

            for col1, col2, corr_val in highly_correlated_pairs:
                st.write(f"- {col1} vs {col2}: {corr_val:.2f}")

            n_pairs = len(highly_correlated_pairs)

            fig_scatter, axes_scatter = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5))

            if n_pairs == 1:
                axes_scatter = [axes_scatter]

            for i, (col1, col2, corr_val) in enumerate(highly_correlated_pairs):
                ax = axes_scatter[i]
                sns.scatterplot(x=df_display[col1], y=df_display[col2], ax=ax, color=sns.color_palette("deep")[i % 10])
                ax.set_title(f'{col1} vs {col2} (Corr: {corr_val:.2f})')
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)

            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            plt.suptitle('Scatter Plots of Highly Correlated Pollutant Pairs', y=1.0, fontsize=16)
            st.pyplot(fig_scatter)
        else:
            st.info("No highly correlated pollutant pairs found with the specified threshold (0.7).")
    else:
        st.warning('Correlation matrix could not be calculated. Ensure numerical data is present.')


elif page == 'Modelling and Prediction':
    st.header('AQI Prediction and Model Evaluation 🔮')
    st.write('This section displays the performance metrics of the trained models (Decision Tree, Random Forest, and KNN) on the test dataset.')

    try:
        target_names_for_reporting = aqi_bucket_label_encoder.classes_
        st.info(f"Detected **{len(target_names_for_reporting)}** unique AQI classes for reporting: **{list(target_names_for_reporting)}**")

    except AttributeError:
        st.error("AQI Bucket Label Encoder could not be loaded or fitted. Cannot display model reports.")
        st.stop()

    model_list = {
        'Decision Tree': dt_classifier,
        'Random Forest': rf_classifier,
        'K-Nearest Neighbors (KNN)': knn_classifier
    }

    st.subheader('Overall Model Accuracy Comparison 📈')

    accuracy_data = {}

    for model_name, model in model_list.items():
        if model is not None:
            try:
                y_pred = model.predict(X_test_app)
                accuracy = accuracy_score(y_test_app, y_pred)
                accuracy_data[model_name] = accuracy
            except Exception as e:
                st.warning(f"Could not calculate accuracy for {model_name}: {e}")

    if accuracy_data:
        classification_metrics_df = pd.DataFrame(accuracy_data.values(), index=accuracy_data.keys(), columns=['Accuracy'])

        fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
        sns.barplot(x=classification_metrics_df.index, y=classification_metrics_df['Accuracy'], palette='viridis', ax=ax_acc)
        ax_acc.set_title('Comparison of Model Accuracy', fontsize=16)
        ax_acc.set_xlabel('Model', fontsize=12)
        ax_acc.set_ylabel('Accuracy', fontsize=12)
        ax_acc.set_ylim(0, 1) 
        ax_acc.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig_acc)
        st.dataframe(classification_metrics_df)
        st.markdown('---')
    else:
        st.warning("No models are currently loaded or trained to compare accuracy.")


    st.subheader('Detailed Model Evaluation Metrics on Test Set')

    for model_name, model in model_list.items():
        st.markdown(f"### {model_name} Performance")

        if model is not None:
            try:
                y_pred = model.predict(X_test_app)

                st.write('#### Classification Report')
                report = classification_report(y_test_app, y_pred, target_names=target_names_for_reporting, output_dict=True)

                report_df = pd.DataFrame(report).transpose().round(2)
                st.dataframe(report_df)

                st.write('#### Confusion Matrix')
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test_app, y_pred)

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=target_names_for_reporting, yticklabels=target_names_for_reporting, ax=ax_cm)
                ax_cm.set_title(f'Confusion Matrix for {model_name}')
                ax_cm.set_ylabel('True Label')
                ax_cm.set_xlabel('Predicted Label')
                

                st.pyplot(fig_cm)

            except Exception as e:
                st.error(f"Error evaluating {model_name}: {e}")
        else:
            st.warning(f"The {model_name} model is not loaded or trained.")

# -------------------------
# Model Visualization Page 
# -------------------------
elif page == 'Model Visualization':
    st.header('Model Visualization: Decision Tree 🌳')
    st.write('Visualizing the structure of the trained Decision Tree for interpretability.')

    if dt_classifier is not None:
        try:
            max_depth = dt_classifier.get_depth()
            
            selected_depth = st.slider(
                'Select Max Depth for Visualization (Controls Image Size/Detail)',
                min_value=1,
                max_value=max_depth,
                value=min(max_depth, 3) 
            )
            
            st.write(f"Displaying Decision Tree up to depth **{selected_depth}** of total depth **{max_depth}**.")
            
            feature_names = X_cols
            class_names = aqi_bucket_label_encoder.classes_.tolist() 
            
            fig_dt, ax_dt = plt.subplots(figsize=(selected_depth * 6, selected_depth * 4)) 

            plot_tree(
                dt_classifier, 
                max_depth=selected_depth, 
                feature_names=feature_names, 
                class_names=class_names, 
                filled=True, 
                rounded=True, 
                proportion=False, 
                fontsize=10,
                ax=ax_dt
            )
            
            ax_dt.set_title(f'Decision Tree Classifier (Max Depth: {selected_depth})', fontsize=20)
            

[Image of decision tree structure]

            
            st.pyplot(fig_dt)
            st.markdown('---')
            st.markdown(f"**Interpretation Note:** Each node shows the splitting condition, the Gini impurity, the number of samples in the node, and the class distribution (`value`). The predicted class (`class`) is the one with the highest count in the `value` list.")
            
        except Exception as e:
            st.error(f"Error visualizing Decision Tree: {e}. Please ensure the model is trained and relevant features exist.")
    else:
        st.warning('The Decision Tree model is not loaded or trained. Please go to the sidebar and check the "Train / retrain models now" option.')
