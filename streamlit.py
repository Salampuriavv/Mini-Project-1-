import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and scalers
json_file = open('model_sigmoid.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_sigmoid.h5")

loaded_scaler_x = load(open('scaler_x.pkl', 'rb'))
loaded_scaler_y = load(open('scaler_y.pkl', 'rb'))

# Function to make predictions
def predict_leakage(input_data):
    input_data = loaded_scaler_x.transform(input_data)
    prediction = loaded_model.predict(input_data)
    prediction_inverse = loaded_scaler_y.inverse_transform(prediction)
    return prediction_inverse

# Function to generate and display visualizations with explanations
def generate_visualizations(df_pressure, predictions):
    # Scatter plot of true vs. predicted values
    fig, axes = plt.subplots(figsize=(12, 8))
    plt.scatter(df_pressure.values, predictions, alpha=0.5)
    plt.title('True vs. Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    st.pyplot(fig)
    st.markdown("The scatter plot above shows the relationship between true and predicted values. This helps assess the model's performance.")

    # Histogram of residuals
    residuals = df_pressure.values - predictions
    fig, axes = plt.subplots(figsize=(12, 8))
    plt.hist(residuals, bins=50)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    st.pyplot(fig)
    st.markdown("The histogram above represents the distribution of residuals (the differences between true and predicted values). A normal distribution indicates good model performance.")

    # Plot accuracy vs. threshold
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc_list = []
    for l in threshold:
        # Your accuracy calculation logic here
        acc_list.append(0.0)  # Replace with your actual accuracy calculation

    fig, axes = plt.subplots(figsize=(12, 8))
    plt.plot(threshold, acc_list, marker='o')
    plt.title('Accuracy vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    st.pyplot(fig)
    st.markdown("The line plot above illustrates how model accuracy changes with different threshold values. Choose a threshold that balances precision and recall based on your application.")

# Streamlit App
def main():
    st.title("Leak Detection Streamlit App")

    # Upload CSV files through Streamlit
    leak_file = st.file_uploader("Choose CSV file for Leak Values", type="csv")
    pressure_file = st.file_uploader("Choose CSV file for Pressure Values", type="csv")

    if leak_file is not None and pressure_file is not None:
        # Read the data
        df_leak = pd.read_csv(leak_file, header=None)
        df_pressure = pd.read_csv(pressure_file, header=None)

        st.write("### Uploaded Leak Values Data")
        st.write(df_leak)

        st.write("### Uploaded Pressure Values Data")
        st.write(df_pressure)

        # Make predictions
        predictions = predict_leakage(df_pressure.values)

        # Display predictions
        st.write("### Predicted Leakage Values")
        st.write(predictions)

        # Generate and display visualizations with explanations
        st.write("### Visualizations and Explanations")
        generate_visualizations(df_pressure, predictions)

if __name__ == "__main__":
    main()
