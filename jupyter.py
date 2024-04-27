import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#read the data
original_data = pd.read_csv("./weatherAUS.csv",na_values=["NA"])

original_data['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
original_data['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)


from sklearn.utils import resample
yes = original_data[original_data['RainTomorrow'] == 1]
no = original_data[original_data['RainTomorrow'] == 0]
yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)
oversampled = pd.concat([no, yes_oversampled])


oversampled['Date'] = oversampled['Date'].fillna(oversampled['Date'].mode()[0])
oversampled['Location'] = oversampled['Location'].fillna(oversampled['Location'].mode()[0])
oversampled['WindGustDir'] = oversampled['WindGustDir'].fillna(oversampled['WindGustDir'].mode()[0])
oversampled['WindDir9am'] = oversampled['WindDir9am'].fillna(oversampled['WindDir9am'].mode()[0])
oversampled['WindDir3pm'] = oversampled['WindDir3pm'].fillna(oversampled['WindDir3pm'].mode()[0])

# Convert categorical features to continuous features with Label Encoding
from sklearn.preprocessing import LabelEncoder
lencoders = {}
for col in oversampled.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    oversampled[col] = lencoders[col].fit_transform(oversampled[col])



# Multiple Imputation by Chained Equations
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
MiceImputed = oversampled.copy(deep=True) 
mice_imputer = IterativeImputer()
MiceImputed.iloc[:, :] = mice_imputer.fit_transform(oversampled)


# Detecting outliers with IQR
Q1 = MiceImputed.quantile(0.25)
Q3 = MiceImputed.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

# Removing outliers from dataset
MiceImputed = MiceImputed[~((MiceImputed < (Q1 - 1.5 * IQR)) |(MiceImputed > (Q3 + 1.5 * IQR))).any(axis=1)]


correlation_matrix = MiceImputed.corr()
print(correlation_matrix)
print()
# mask = np.triu(np.ones_like(correlation_matrix, dtype=np.bool))

# Standardizing data
from sklearn import preprocessing
r_scaler = preprocessing.MinMaxScaler()
r_scaler.fit(MiceImputed)
modified_data = pd.DataFrame(r_scaler.transform(MiceImputed), index=MiceImputed.index, columns=MiceImputed.columns)

# Feature Importance using Filter Method (Chi-Square)
from sklearn.feature_selection import SelectKBest, chi2
X = modified_data.loc[:,modified_data.columns!='RainTomorrow']
y = modified_data[['RainTomorrow']]
selector = SelectKBest(chi2, k=10)
selector.fit(X, y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)])

features = modified_data[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']]
target = modified_data['RainTomorrow']

# spliting the data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=123)




import os
import joblib  # For saving models
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Define function to train model, return performance metrics, and save model
def train_and_save_model(model, X_train, y_train, X_test, y_test, model_name, model_path):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    coh_kap = cohen_kappa_score(y_test, y_pred)
    cf_matrix = confusion_matrix(y_test, y_pred)

    # Print metrics
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"ROC-AUC: {roc_auc:.2%}")
    print(f"Cohen's Kappa: {coh_kap:.2%}")
    print("Confusion Matrix:")
    print(cf_matrix)
    
    # Save the model to the specified path
    model_filename = os.path.join(model_path, f"{model_name}.pkl")
    joblib.dump(model, model_filename)
    print(f"Model saved at: {model_filename}")
    
    return model

# Create the directory for storing models if it doesn't exist
model_path = "/mnt/codebox/python/project/models/"
os.makedirs(model_path, exist_ok=True)

# Train and compile multiple models
# 1. Support Vector Machine (SVM)
svc_model = svm.SVC(kernel="linear")  # You can customize the kernel
train_and_save_model(svc_model, X_train, y_train, X_test, y_test, "Support_Vector_Machine", model_path)

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=12345)
train_and_save_model(rf_model, X_train, y_train, X_test, y_test, "Random_Forest", model_path)

# 3. Gaussian Naive Bayes
gnb_model = GaussianNB()
train_and_save_model(gnb_model, X_train, y_train, X_test, y_test, "Gaussian_Naive_Bayes", model_path)

# 4. XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=16, random_state=12345)
train_and_save_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost", model_path)

