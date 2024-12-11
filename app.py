import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_models():
    """Load saved Random Forest model and scaler"""
    try:
        classifier = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return classifier, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_input(input_data, label_encoders, scaler):
    """
    Preprocess input data to match training features and encode categorical variables
    """
    # Define expected columns
    expected_columns = ['studytime', 'absences', 'G1', 'G2', 'age', 'famsize', 'traveltime', 'failures', 'schoolsup', 'higher']

    # Fill missing values with default values
    default_values = {
        'studytime': 2, 'absences': 0, 'G1': 10, 'G2': 10, 'age': 18,
        'famsize': 'GT3', 'traveltime': 1, 'failures': 0, 'schoolsup': 'no', 'higher': 'yes'
    }

    processed_data = {col: input_data.get(col, default_values[col]) for col in expected_columns}

    # Encode categorical variables
    for col in ['famsize', 'schoolsup', 'higher']:
        if col in processed_data:
            processed_data[col] = label_encoders[col].transform([processed_data[col]])[0]

    # Convert to DataFrame
    input_df = pd.DataFrame([processed_data])

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    return input_scaled

def predict_student_grade(input_data, classifier, scaler, label_encoders):
    """Make predictions using the loaded Random Forest model"""
    try:
        # Preprocess input data
        input_scaled = preprocess_input(input_data, label_encoders, scaler)

        # Make predictions
        prediction_class = classifier.predict(input_scaled)[0]
        proba = classifier.predict_proba(input_scaled)[0]

        return {
            'pass_fail': 'Pass' if prediction_class == 1 else 'Fail',
            'pass_probability': round(proba[1] * 100, 2)
        }

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    st.title("Student Grade Prediction")

    # Load model and scaler
    classifier, scaler = load_models()
    if not classifier:
        return

    # Initialize LabelEncoders for categorical variables
    label_encoders = {
        'famsize': LabelEncoder().fit(['LE3', 'GT3']),
        'schoolsup': LabelEncoder().fit(['no', 'yes']),
        'higher': LabelEncoder().fit(['no', 'yes'])
    }

    # Input fields for the 10 features
    input_data = {}
    input_data['studytime'] = st.sidebar.slider("Study Time (1-4)", min_value=1, max_value=4, value=2)
    input_data['absences'] = st.sidebar.number_input("Number of Absences (0-93)", min_value=0, max_value=93, value=0)
    input_data['G1'] = st.sidebar.number_input("G1 Grade (0-20)", min_value=0, max_value=20, value=10)
    input_data['G2'] = st.sidebar.number_input("G2 Grade (0-20)", min_value=0, max_value=20, value=10)
    input_data['age'] = st.sidebar.number_input("Age (15-22)", min_value=15, max_value=22, value=18)
    input_data['famsize'] = st.sidebar.selectbox("Family Size", ['LE3', 'GT3'])
    input_data['traveltime'] = st.sidebar.slider("Travel Time (1-4)", min_value=1, max_value=4, value=2)
    input_data['failures'] = st.sidebar.number_input("Number of Failures (0-4)", min_value=0, max_value=4, value=0)
    input_data['schoolsup'] = st.sidebar.selectbox("School Support", ['yes', 'no'])
    input_data['higher'] = st.sidebar.selectbox("Desire for Higher Education", ['yes', 'no'])

    # Prediction button
    if st.sidebar.button("Predict Grade"):
        if classifier:
            # Make prediction
            result = predict_student_grade(input_data, classifier, scaler, label_encoders)

            if result:
                # Display results
                st.header("Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Pass/Fail", result['pass_fail'])

                with col2:
                    st.metric("Passing Probability", f"{result['pass_probability']}%")

                # Additional insights
                st.subheader("Insights")
                if result['pass_fail'] == 'Pass':
                    st.success("Great job! Keep up the good work!")
                else:
                    st.warning("You might need additional support. Consider studying more or seeking help.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Powered by Machine Learning")

if __name__ == "__main__":
    main()
