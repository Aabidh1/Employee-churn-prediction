import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(page_title="Employee Churn Prediction", layout="wide", page_icon="📊")

# Add custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f4f6f9;
            color: #333;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
        }
        .stSlider > div {
            background-color: #e8f5e9;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Employee Churn Prediction")
st.sidebar.markdown("Use this app to predict employee churn based on input data.")
app_selection = st.sidebar.radio("Select App", ["Single Prediction", "Prediction Using Test File"])

# Function to load models
def load_model(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# Function to process and predict single entry
def show_prediction(pipeline, inputs):
    sample = pd.DataFrame([inputs])
    result = pipeline.predict(sample)
    if result[0] == 1:
        st.markdown("### 🔴 An employee may **leave** the organization.")
    else:
        st.markdown("### 🟢 An employee may **stay** with the organization.")

# Single Prediction App
if app_selection == "Single Prediction":
    st.title("Single Employee Churn Prediction")

    # Load the pipeline
    pipeline = load_model('pipeline.pkl')

    # Employee data input fields
    st.markdown("### Input Employee Details:")
    p1 = st.slider("Employee satisfaction level", 0.0, 1.0, 0.5)
    p2 = st.slider("Last evaluation score", 0.0, 1.0, 0.5)
    p3 = st.slider("Number of projects assigned to", 1, 10, 5)
    p4 = st.slider("Average monthly hours worked", 50, 300, 150)
    p5 = st.slider("Time spent at the company (years)", 1, 10, 3)
    p6 = st.radio("Had a work accident?", [0, 1])
    p7 = st.radio("Promotion in the last 5 years?", [0, 1])

    options = ('sales', 'technical', 'support', 'IT', 'product_mng', 'marketing',
               'RandD', 'accounting', 'hr', 'management')
    p8 = st.selectbox("Department", options)

    options1 = ('low', 'medium', 'high')
    p9 = st.selectbox("Salary category", options1)

    # Predict button
    if st.button("Predict"):
        inputs = {
            'satisfaction_level': p1,
            'last_evaluation': p2,
            'number_project': p3,
            'average_montly_hours': p4,
            'time_spend_company': p5,
            'Work_accident': p6,
            'promotion_last_5years': p7,
            'departments': p8,
            'salary': p9
        }
        show_prediction(pipeline, inputs)

# Bulk Prediction App
else:
    st.title("Bulk Employee Churn Prediction")

    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load the CSV file
            data = pd.read_csv(uploaded_file)

            # Clean column names and remove duplicates
            data.columns = data.columns.str.strip()
            data.rename(columns={'Departments ': 'departments'}, inplace=True)
            data = data.drop_duplicates()

            # Load the pipeline
            pipeline = load_model('pipeline.pkl')

            # Predict results
            results = pipeline.predict(data)

            # Add predictions to the data
            data['Predicted_target'] = ["Leave" if res == 1 else "Stay" for res in results]

            # Display and save processed data
            st.write("### Processed Data:")
            st.dataframe(data)

            # Save to a CSV file
            processed_file_name = 'processed_data.csv'
            data.to_csv(processed_file_name, index=False)
            st.success(f"Processed data saved as `{processed_file_name}`")

        except Exception as e:
            st.error(f"Error processing file: {e}")
