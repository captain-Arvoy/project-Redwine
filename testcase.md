Test Cases for Data Processing
1. Data Loading

    Test Case: Ensure the dataset is loaded correctly.
    Expected Outcome: The data is loaded without errors, and the number of rows and columns matches expectations.
    Validation: Check that original_data.shape returns the correct dimensions, and there are no unexpected missing values.

2. Data Preprocessing

    Test Case: Validate that categorical values are replaced correctly.
    Expected Outcome: All 'RainToday' and 'RainTomorrow' values are converted to 0 or 1.
    Validation: Check original_data['RainToday'].unique() and original_data['RainTomorrow'].unique() to ensure they contain only 0 and 1.

3. Imputation

    Test Case: Verify that missing values are imputed properly.
    Expected Outcome: No NaNs in the MiceImputed DataFrame after imputation.
    Validation: Check MiceImputed.isna().sum() to ensure all missing values are handled.

4. Outlier Detection and Removal

    Test Case: Ensure outliers are detected and removed correctly.
    Expected Outcome: The resulting dataset has fewer rows after removing outliers.
    Validation: Check the shape of MiceImputed before and after outlier removal.

5. Standardization

    Test Case: Confirm that data is standardized properly.
    Expected Outcome: All numerical features are scaled to a consistent range (e.g., 0-1).
    Validation: Check that the min and max values of modified_data are within the expected range.

6. Feature Selection

    Test Case: Validate that the correct features are selected.
    Expected Outcome: The selected features match the intended list of top 10 features.
    Validation: Verify the features selected by SelectKBest match the expected ones.

Test Cases for Model Building and Evaluation
7. Train-Test Split

    Test Case: Validate the split between training and testing data.
    Expected Outcome: The training and testing datasets contain the correct number of samples.
    Validation: Check X_train.shape, X_test.shape, y_train.shape, and y_test.shape to ensure they contain the expected number of samples.

8. Model Training

    Test Case: Ensure models are trained without errors.
    Expected Outcome: All models complete training successfully.
    Validation: Check that run_model completes without exceptions for each model.

9. Model Performance

    Test Case: Validate model accuracy and other metrics.
    Expected Outcome: Models achieve reasonable accuracy and other metrics like ROC-AUC and Cohen's Kappa.
    Validation: Check that accuracy, ROC-AUC, and Cohen's Kappa are within expected ranges for each model.

10. Confusion Matrix and Classification Report

    Test Case: Ensure confusion matrix and classification report are generated correctly.
    Expected Outcome: Confusion matrix has correct dimensions, and classification report provides accurate results.
    Validation: Check that classification_report and confusion_matrix do not throw errors and provide meaningful results.

11. Model Comparison

    Test Case: Validate the model comparison results.
    Expected Outcome: The comparison plot accurately represents model performance.
    Validation: Ensure the comparison plot is generated without errors and shows accuracy and time taken for each model.
