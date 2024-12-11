import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load models with specific exception handling for compatibility
def load_models():
    """Load saved Random Forest model and scaler"""
    try:
        # Explicitly set the paths relative to the current directory
        classifier = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')  # Load the scaler if used
        return classifier, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_input(input_data, expected_columns):
    """
    Preprocess input data to match training features and encode categorical variables
    """
    # Default values for all features
    default_values = {col: 0 for col in expected_columns}

    # Fill missing values with defaults
    processed_data = {col: input_data.get(col, default_values[col]) for col in expected_columns}

    # Convert to DataFrame
    input_df = pd.DataFrame([processed_data])

    return input_df


def predict_student_grade(input_data, classifier, scaler):
    """Make predictions using the loaded Random Forest model"""
    try:
        # Get the column names used during training
        expected_columns = classifier.feature_names_in_

        # Preprocess input data to match training features
        input_df = preprocess_input(input_data, expected_columns)

        # Scale data if a scaler is available
        if scaler:
            input_df = scaler.transform(input_df)

        # Ensure column order matches the model
        input_df = input_df[expected_columns]

        # Make predictions
        prediction_class = classifier.predict(input_df)[0]
        proba = classifier.predict_proba(input_df)[0]

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

    # Input fields for the 10 features
    input_data = {}

    # Numeric and categorical inputs
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
            result = predict_student_grade(input_data, classifier, scaler)

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
