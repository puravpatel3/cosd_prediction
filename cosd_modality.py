import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import gzip
import shutil
from io import BytesIO
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Download the compressed model from GitHub
@st.cache_resource
def download_and_load_model():
    model_url = 'https://github.com/puravpatel3/cosd_prediction/raw/main/rf_model_modality.pkl.gz'
    response = requests.get(model_url)
    compressed_file = BytesIO(response.content)

    # Decompress the model
    with gzip.open(compressed_file, 'rb') as f_in:
        model = joblib.load(f_in)
    
    return model

# Load the model
model = download_and_load_model()

# Load the dataset to get available options and historical averages
data_file = r'https://github.com/puravpatel3/cosd_prediction/raw/main/cosdpos_ci_rf_modality.csv'
df = pd.read_csv(data_file)

# Normalize column names to avoid case-sensitivity issues
df.columns = df.columns.str.lower()

# Set the title of the app
st.title('COSD Prediction Model')

# User input for Eset Creation Date
eset_creation_date = st.date_input("Eset Creation Date", value=datetime.today())

# User input for categorical variables
modality = st.selectbox('Modality', df['modality'].unique())
country = st.selectbox('Country', df['country'].unique())
customer_class_new = st.selectbox('Customer Class New', df['customer class new'].unique())

# Calculate average lead times from historical data based on user inputs
def estimate_lead_times(modality, country, customer_class_new):
    filtered_df = df[
        (df['modality'] == modality) &
        (df['country'] == country) &
        (df['customer class new'] == customer_class_new)
    ]
    
    # Calculate average lead times
    avg_aosd = filtered_df['lead_time_from_ecd_to_aosd'].mean()
    avg_asdd = filtered_df['lead_time_from_ecd_to_asdd'].mean()
    avg_asds = filtered_df['lead_time_from_ecd_to_asds'].mean()
    avg_procure_date = filtered_df['lead_time_from_ecd_to_procure_date'].mean()
    
    # Return the average values (use default if no data is available)
    return (
        avg_aosd if not np.isnan(avg_aosd) else 30,
        avg_asdd if not np.isnan(avg_asdd) else 30,
        avg_asds if not np.isnan(avg_asds) else 30,
        avg_procure_date if not np.isnan(avg_procure_date) else 30
    )

# Estimate lead times based on user inputs
estimated_aosd, estimated_asdd, estimated_asds, estimated_procure_date = estimate_lead_times(modality, country, customer_class_new)

# Display the estimated lead times
st.write(f"Estimated Lead Time from ECD to AOSD (days): {estimated_aosd}")
st.write(f"Estimated Lead Time from ECD to ASDD (days): {estimated_asdd}")
st.write(f"Estimated Lead Time from ECD to ASDS (days): {estimated_asds}")
st.write(f"Estimated Lead Time from ECD to Procure Date (days): {estimated_procure_date}")

# Encode categorical variables using the same encoding as the model
le_modality = LabelEncoder()
df['modality encoded'] = le_modality.fit_transform(df['modality'])
le_country = LabelEncoder()
df['country encoded'] = le_country.fit_transform(df['country'])
le_customer_class = LabelEncoder()
df['customer class new encoded'] = le_customer_class.fit_transform(df['customer class new'])

# Transform user input
modality_encoded = le_modality.transform([modality])[0]
country_encoded = le_country.transform([country])[0]
customer_class_new_encoded = le_customer_class.transform([customer_class_new])[0]

# Create input array for prediction
input_data = np.array([[estimated_aosd, estimated_asdd, estimated_asds, estimated_procure_date,
                        modality_encoded, country_encoded, customer_class_new_encoded]])

# Perform prediction using the model
if st.button('Predict COSD'):
    predicted_lead_time = model.predict(input_data)
    
    # Calculate confidence intervals (if available)
    y_std_dev = 1.96 * np.std(predicted_lead_time)  # Placeholder, adjust based on model
    confidence_interval_lower = predicted_lead_time - y_std_dev
    confidence_interval_upper = predicted_lead_time + y_std_dev

    # Calculate predicted COSD by adding lead time to Eset Creation Date
    predicted_cosd_date = pd.to_datetime(eset_creation_date) + pd.to_timedelta(predicted_lead_time, unit='D')
    confidence_lower_date = pd.to_datetime(eset_creation_date) + pd.to_timedelta(confidence_interval_lower, unit='D')
    confidence_upper_date = pd.to_datetime(eset_creation_date) + pd.to_timedelta(confidence_interval_upper, unit='D')

    # Output results to the user
    st.write(f"**Predicted COSD**: {predicted_cosd_date[0].date()}")
    st.write(f"**Confidence Interval Lower**: {confidence_lower_date[0].date()}")
    st.write(f"**Confidence Interval Upper**: {confidence_upper_date[0].date()}")
