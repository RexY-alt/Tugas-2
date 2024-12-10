import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Fungsi untuk mempersiapkan data
def load_and_prepare_data():
    # Load dataset
    data = pd.read_csv('student-mat.csv')
    
    # Prepare features
    X = data[['studytime', 'absences', 'G1', 'G2', 'age', 'famsize', 'traveltime', 'failures', 'schoolsup', 'higher']]
    y_classification = (data['G3'] >= 13).astype(int)  # Binary target for classification
    y_regression = data['G3']  # Continuous target for regression

    # Encode categorical variables if needed
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    return X, y_classification, y_regression

# Fungsi untuk melatih dan menyimpan model
def train_and_save_models(force_retrain=False):
    """
    Train models and save them. If models already exist, load them unless force_retrain is True.
    
    Args:
        force_retrain (bool): If True, will retrain models even if they exist
    
    Returns:
        tuple: Classifier and Regressor models
    """
    # Paths for saving models
    classifier_path = 'student_classifier_model.pkl'
    regressor_path = 'student_regressor_model.pkl'
    
    # Check if models already exist and should not be retrained
    if not force_retrain and os.path.exists(classifier_path) and os.path.exists(regressor_path):
        st.info("Loading existing models...")
        classifier = joblib.load(classifier_path)
        regressor = joblib.load(regressor_path)
        return classifier, regressor
    
    # Prepare data
    X, y_classification, y_regression = load_and_prepare_data()
    
    # Split data
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train_class, y_train_class)

    # Train Random Forest Regressor
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train_reg, y_train_reg)
    
    # Save models
    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)
    
    st.info("Models trained and saved successfully.")
    return classifier, regressor

# Main Streamlit app
def main():
    st.title('Student Grade Prediction')
    
    # Train or load models
    try:
        classifier, regressor = train_and_save_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure 'student-mat.csv' is in the correct directory.")
        return
    
    # Sidebar for input
    st.sidebar.header('Student Information Input')
    
    # Input fields with default values and validation
    studytime = st.sidebar.slider('Study Time', min_value=1, max_value=4, value=2, help='Hours of study time per week')
    absences = st.sidebar.number_input('Absences', min_value=0, max_value=93, value=0, help='Number of school absences')
    G1 = st.sidebar.number_input('G1 Grade', min_value=0, max_value=20, value=10, help='First period grade')
    G2 = st.sidebar.number_input('G2 Grade', min_value=0, max_value=20, value=10, help='Second period grade')
    age = st.sidebar.slider('Age', min_value=15, max_value=22, value=18, help='Student age')
    famsize = st.sidebar.selectbox('Family Size', [('GT3', 1), ('LE3', 0)], format_func=lambda x: x[0], help='Family size')
    traveltime = st.sidebar.slider('Travel Time', min_value=1, max_value=4, value=2, help='Home to school travel time')
    failures = st.sidebar.number_input('Previous Failures', min_value=0, max_value=4, value=0, help='Number of past class failures')
    schoolsup = st.sidebar.selectbox('School Support', [('Yes', 1), ('No', 0)], format_func=lambda x: x[0], help='Extra educational support')
    higher = st.sidebar.selectbox('Want Higher Education', [('Yes', 1), ('No', 0)], format_func=lambda x: x[0], help='Desire for higher education')
    
    # Prepare user input
    user_input = pd.DataFrame({
        'studytime': [studytime],
        'absences': [absences],
        'G1': [G1],
        'G2': [G2],
        'age': [age],
        'famsize': [famsize[1]],
        'traveltime': [traveltime],
        'failures': [failures],
        'schoolsup': [schoolsup[1]],
        'higher': [higher[1]]
    })
    
    # Prediction section
    st.header('Prediction Results')
    
    # Classification prediction (Pass/Fail)
    prediction_class = classifier.predict(user_input)[0]
    class_proba = classifier.predict_proba(user_input)[0]
    
    # Regression prediction (G3 grade)
    prediction_reg = regressor.predict(user_input)[0]
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric('Pass/Fail Prediction', 'Pass' if prediction_class == 1 else 'Fail')
        st.metric('Probability of Passing', f'{class_proba[1]:.2%}')
    
    with col2:
        st.metric('Predicted Final Grade (G3)', f'{prediction_reg:.2f}')
    
    # Optional: Feature importance visualization
    st.header('Model Insights')
    st.text('Feature importances for predicting student performance')
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': ['studytime', 'absences', 'G1', 'G2', 'age', 'famsize', 'traveltime', 'failures', 'schoolsup', 'higher'],
        'importance': regressor.feature_importances_
    }).sort_values('importance', ascending=False)
    
    st.bar_chart(feature_importance.set_index('feature'))

# Tambahkan requirements.txt berikut
def create_requirements():
    requirements = """
streamlit
pandas
numpy
scikit-learn
joblib
    """
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())

if __name__ == '__main__':
    create_requirements()  # Buat file requirements.txt
    main()
