import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import OrdinalEncoder
import math

# Load the trained model
model_path = 'https://github.com/puravpatel3/cosd_prediction/raw/main/rf_model_modality.pkl.gz'
model = joblib.load(model_path)

# Load the dataset to get available options and historical averages
data_file = 'https://github.com/puravpatel3/cosd_prediction/raw/main/cosdpos_ci_rf_modality.csv'
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

# Function to estimate lead times from historical data based on user inputs
def estimate_lead_times(modality, country, customer_class_new):
    filtered_df = df[
        (df['modality'] == modality) &
        (df['country'] == country) &
        (df['customer class new'] == customer_class_new)
    ]
    
    # Calculate average lead times and round them up
    avg_aosd = math.ceil(filtered_df['lead_time_from_ecd_to_aosd'].mean())
    avg_asdd = math.ceil(filtered_df['lead_time_from_ecd_to_asdd'].mean())
    avg_asds = math.ceil(filtered_df['lead_time_from_ecd_to_asds'].mean())
    avg_procure_date = math.ceil(filtered_df['lead_time_from_ecd_to_procure_date'].mean())
    
    return (
        avg_aosd if avg_aosd is not None else 30,
        avg_asdd if avg_asdd is not None else 30,
        avg_asds if avg_asds is not None else 30,
        avg_procure_date if avg_procure_date is not None else 30
    )

# Estimate lead times based on user inputs
estimated_aosd, estimated_asdd, estimated_asds, estimated_procure_date = estimate_lead_times(modality, country, customer_class_new)

# Calculate the estimated dates
estimated_dates = [
    eset_creation_date + timedelta(days=estimated_aosd),
    eset_creation_date + timedelta(days=estimated_asdd),
    eset_creation_date + timedelta(days=estimated_asds),
    eset_creation_date + timedelta(days=estimated_procure_date)
]

# Create a DataFrame for the Estimated Lead Time Table
lead_time_data = {
    'Description': [
        "Estimated Lead Time from ECD to AOSD",
        "Estimated Lead Time from ECD to ASDD",
        "Estimated Lead Time from ECD to ASDS",
        "Estimated Lead Time from ECD to Procure Date"
    ],
    'Number of Days': [estimated_aosd, estimated_asdd, estimated_asds, estimated_procure_date],
    'Estimated Date': [d.strftime("%Y-%m-%d") for d in estimated_dates]
}

lead_time_table = pd.DataFrame(lead_time_data)

# Display the estimated lead times as a table
st.subheader("Estimated Lead Times")
st.table(lead_time_table)

# Ordinal encoding of categorical variables
encoder = OrdinalEncoder()
df[['modality', 'country', 'customer class new']] = encoder.fit_transform(df[['modality', 'country', 'customer class new']])

# Transform user input
input_data = encoder.transform([[modality, country, customer_class_new]])

# Create input array for prediction
input_array = [[estimated_aosd, estimated_asdd, estimated_asds, estimated_procure_date] + list(input_data[0])]

# Perform prediction using the model
if st.button('Predict COSD'):
    predicted_lead_time = model.predict(input_array)
    
    # Calculate predicted COSD by adding lead time to Eset Creation Date
    predicted_cosd_date = pd.to_datetime(eset_creation_date) + pd.to_timedelta(predicted_lead_time, unit='D')

    # Output results to the user
    st.write(f"**Predicted COSD**: {predicted_cosd_date[0].date()}")
