from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
nav = st.sidebar.radio("Navigation",["Home","EDA","Model metrics","model comparision","test and predict"])
def load_data(nrows):
    data = pd.read_csv(DATA_URL,nrows=nrows)
    lowercase = lambda x: str(x).lower()
    return data
def handleMissingData(df):
    missing_threshold = 0.3  # 30% missing data
    drop_columns = [col for col in df.columns if df[col].isna().mean() > missing_threshold]
    df.drop(columns=drop_columns, inplace=True)
    
    categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    numerical_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
                      'Humidity9am', 'Humidity3pm', 'Temp9am', 'Temp3pm']
    
    for col in numerical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)
def handleCategoricalData(df):
    from sklearn.preprocessing import LabelEncoder
    lencoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        lencoders[col] = LabelEncoder()
        df[col] = lencoders[col].fit_transform(df[col])
def pipeline(df):
    handleMissingData(df)    
    handleCategoricalData(df)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
if nav == "Home":
    st.write("Home")
if nav == "EDA":
    st.write("EDA")


    st.title("Exploratory Data Analysis (EDA) for Weather Dataset")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        pipeline(df)
        st.write("pipeline performed")
        correlation_matrix = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()
        
        # Histogram for a specific numerical feature (e.g., Rainfall)
        plt.figure()
        sns.histplot(df['Rainfall'], kde=True)
        plt.title("Rainfall Distribution")
        plt.show()
        
        # Scatter Plot for two numerical features (e.g., Temp3pm vs. Rainfall)
        plt.figure()
        sns.scatterplot(data=df, x='Temp3pm', y='Rainfall', hue='RainTomorrow')
        plt.title("Scatter Plot: Temp3pm vs. Rainfall")
        plt.show()
    
    else:
        st.warning("Please upload a CSV file to visualize data.")
if nav == "Model metrics":

    metrics = {
        "Model": ["Logistic Regression", "SVM", "Random Forest", "Naive Bayes", "XGBoost"],
        "Accuracy": [0.7951030699028782, 0.7950853145363186, 0.9261554304788623, 0.7548871646455141, 0.9567124163278351],
        "ROC_AUC": [0.7886715626927191, 0.7882681572132252, 0.9276503804181361, 0.7523949817666005, 0.958126302551973],
        "Cohen_Kappa": [0.5810409951349718, 0.5806794342918455, 0.8507644353347799, 0.5033855289543485, 0.9124197647096522],
        "Time_taken": [22.531682014465332, 10.262767791748047, 36.748751401901245, 0.16623449325561523, 15.112218379974365],
    }
    
    confusion_matrices = {

        "Logistic Regression": [[26615, 5064], [6474, 18168]],
        "SVM": [[26700, 4979], [6562, 18080]],
        "Random Forest": [[29008, 2671], [1488, 23154]],
        "Naive Bayes": [[24467, 7212],[6593, 18049]],
        "XGBoost": [[29994, 1685], [753, 23889]],
    }
    
    metrics_df = pd.DataFrame(metrics)
    
    st.title("Model Metrics for Rainfall Prediction")
    
    selected_model = st.selectbox(
        "Select a model to view its confusion matrix",
        metrics_df["Model"]
    )
    
    
    st.write(f"Metrics for {selected_model}")
    model_data = metrics_df[metrics_df["Model"] == selected_model]
    st.dataframe(model_data)
    
    st.write(f"Confusion Matrix for {selected_model}")
    conf_matrix = confusion_matrices[selected_model]
    
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {selected_model}")
    st.pyplot(fig)
    
    csv_data = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Metrics as CSV",
        data=csv_data,
        file_name='model_metrics.csv',
        mime='text/csv',
    )
if nav == 'model comparision':
    
    model_metrics = {
        "Model": ["Logistic Regression", "Support Vector Machine", "Random Forest", "Gaussian Naive Bayes", "XGBoost"],
        "Accuracy": [79.512083, 79.508531, 92.615543, 75.488716, 95.671242],
        "ROC_AUC": [0.788692, 0.788268, 0.927650, 0.752395, 0.958126],
        "Cohen_Kappa": [0.581079, 0.580679, 0.850764, 0.503386, 0.912420],
        "Time_taken": [22.639017, 10.912878, 36.906135, 0.162580, 13.780446],
        "Accuracy Time Ratio": [3.512170, 7.285753, 2.509489, 464.315961, 6.942536],
    }
    
    df = pd.DataFrame(model_metrics)
    
    st.title("Model Metrics Visualization")
    
    selected_metric = st.selectbox(
        "Select a metric to visualize",
        ["Accuracy", "ROC_AUC", "Cohen_Kappa", "Time_taken", "Accuracy Time Ratio"]
    )
    
    st.write(f"Bar Plot for {selected_metric} across Models")
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y=selected_metric, data=df, ax=ax)
    ax.set_xlabel("Model")
    ax.set_ylabel(selected_metric)
    ax.set_title(f"{selected_metric} across Models")
    st.pyplot(fig)
if nav == 'test and predict':
        
    # Define the directory where pre-trained models are stored
    model_dir = "./models/"
    
    # Define a function to load the selected model
    def load_model(model_name):
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
    
    # Function to preprocess the data and ensure it has the correct features
    def preprocess_data(dataframe, label_encoders):
        # Convert categorical features to numeric
        categorical_columns = ["Location"]  # List of categorical columns to encode
        for col in categorical_columns:
            if col in dataframe.columns:
                if col not in label_encoders:
                    label_encoders[col] = LabelEncoder()  # Create a new encoder
                # Fit and transform the categorical column
                dataframe[col] = label_encoders[col].fit_transform(dataframe[col])
        
        # Drop any unnecessary columns (e.g., Date)
        columns_to_drop = ["Date"]  # Add any other irrelevant columns
        dataframe.drop(columns=columns_to_drop, errors='ignore', inplace=True)
        
        return dataframe
    
    # Streamlit app with 'test and predict' route
    st.title("Validate models")
    uploaded_file = st.file_uploader("Upload a CSV file with test data", type=["csv"])
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(test_data.head())
    
        # Step 2: Model selection dropdown
        model_choice = st.selectbox(
            "Select a model to predict rainfall",
            ["Logistic Regression", "Support Vector Machine", "Random Forest", "Gaussian Naive Bayes", "XGBoost"]
        )
    
        try:
            # Step 3: Load the selected model
            model = load_model(model_choice)
    
            # Step 4: Preprocess the test data
            label_encoders = {}  # Label encoders for categorical data
            test_data = preprocess_data(test_data, label_encoders)
    
            # Step 5: Ensure the test data has the correct features
            expected_features = ["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustDir", 
                                 "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", 
                                 "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", 
                                 "Temp9am", "Temp3pm", "RainToday"]
            test_data = test_data[expected_features]  # Keep only the relevant features
    
            # Step 6: Make predictions with the loaded model
            predictions = model.predict(test_data)
    
            # Step 7: Display predictions
            prediction_results = test_data.copy()  # Create a copy of the test data
            prediction_results["Prediction"] = predictions
            prediction_results["RainPrediction"] = prediction_results["Prediction"].apply(
                lambda x: "Rain" if x == 1 else "No Rain"
            )
            
            st.write("Prediction Results:")
            st.dataframe(prediction_results)  # Display the prediction results
    
            # Optional: Allow users to download the predictions
            if st.button("Download Predictions CSV"):
                prediction_results.to_csv("rainfall_predictions.csv", index=False)
                st.download_button(
                    label="Download Predictions",
                    data=prediction_results.to_csv(index=False),
                    file_name="rainfall_predictions.csv",
                    mime="text/csv"
                )
    
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload dataset to proceed.")


    




