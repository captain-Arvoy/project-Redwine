import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
nav = st.sidebar.radio("Navigation",["Home","EDA","Model metrics","model comparision"])
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


    




