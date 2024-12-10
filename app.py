import streamlit as st
import joblib
import pandas as pd
import numpy as np

def load_models():
    """Load saved models and scaler"""
    try:
        classifier = joblib.load('classifier_model.pkl')
        regressor = joblib.load('regressor_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return classifier, regressor, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def predict_student_grade(input_data, classifier, regressor, scaler):
    """Make predictions using loaded models"""
    try:
        # Define all features used during model training (make sure to add all features needed)
        required_features = [
            'studytime', 'absences', 'G1', 'G2', 'age', 'famsize', 'traveltime', 'failures', 
            'schoolsup', 'higher', 'Dalc', 'Fedu', 'Fjob', 'Medu', 'Mjob'
        ]
        
        # Add missing features to input_data with default values (e.g., 0 or 'None' if categorical)
        for feature in required_features:
            if feature not in input_data:
                input_data[feature] = 0  # Default value for missing features (0 for numeric, 'None' for categorical)

        # Prepare input data (make sure it's in the same order as the training data)
        input_df = pd.DataFrame([input_data])[required_features]

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make predictions
        prediction_class = classifier.predict(input_scaled)[0]
        prediction_reg = regressor.predict(input_scaled)[0]
        proba = classifier.predict_proba(input_scaled)[0]

        return {
            'pass_fail': 'Pass' if prediction_class == 1 else 'Fail',
            'predicted_grade': round(prediction_reg, 2),
            'pass_probability': round(proba[1] * 100, 2)
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    # Set page title and favicon
    st.set_page_config(page_title="Student Grade Predictor", page_icon=":student:")

    # Load models
    classifier, regressor, scaler = load_models()
    
    # Main title
    st.title("🎓 Student Grade Prediction")
    st.write("Predict student performance using machine learning")

    # Sidebar for user inputs
    st.sidebar.header("Input Student Information")
    
    # Input fields (Only 10 user-input features)
    input_data = {}
    input_data['studytime'] = st.sidebar.slider("Study Time (1-4)", min_value=1, max_value=4, value=2)
    input_data['absences'] = st.sidebar.number_input("Number of Absences (0-93)", min_value=0, max_value=93, value=0)
    input_data['G1'] = st.sidebar.number_input("G1 Grade (0-20)", min_value=0, max_value=20, value=10)
    input_data['G2'] = st.sidebar.number_input("G2 Grade (0-20)", min_value=0, max_value=20, value=10)
    input_data['age'] = st.sidebar.number_input("Age (15-22)", min_value=15, max_value=22, value=18)
    input_data['famsize'] = st.sidebar.selectbox("Family Size", [0, 1], format_func=lambda x: "LE3" if x == 0 else "GT3")
    input_data['traveltime'] = st.sidebar.slider("Travel Time (1-4)", min_value=1, max_value=4, value=2)
    input_data['failures'] = st.sidebar.number_input("Number of Failures (0-4)", min_value=0, max_value=4, value=0)
    input_data['schoolsup'] = st.sidebar.selectbox("School Support", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    input_data['higher'] = st.sidebar.selectbox("Higher Education Desire", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Prediction button
    if st.sidebar.button("Predict Grade"):
        if classifier and regressor and scaler:
            # Make prediction
            result = predict_student_grade(input_data, classifier, regressor, scaler)
            
            if result:
                # Display results in main area
                st.header("Prediction Results")
                
                # Columns for better layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Pass/Fail", result['pass_fail'])
                
                with col2:
                    st.metric("Predicted Grade", result['predicted_grade'])
                
                with col3:
                    st.metric("Passing Probability", f"{result['pass_probability']}%")
                
                # Additional insights
                st.subheader("Insights")
                if result['pass_fail'] == 'Pass':
                    st.success("Great job! Keep up the good work!")
                else:
                    st.warning("You might need additional support. Consider studying more or seeking help.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Machine Learning Student Grade Predictor")

if __name__ == "__main__":
    main()
