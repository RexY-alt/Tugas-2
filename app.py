import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def load_models():
    """Load saved Random Forest model and scaler"""
    try:
        classifier = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')  # Load the scaler if used
        return classifier, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_input(input_data):
    """Preprocess input data to match training features and encode categorical variables"""
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    
    # Convert categorical variables (use LabelEncoder)
    input_data['famsize'] = label_encoder.fit_transform([input_data['famsize']])[0]
    input_data['schoolsup'] = label_encoder.fit_transform([input_data['schoolsup']])[0]
    input_data['higher'] = label_encoder.fit_transform([input_data['higher']])[0]
    
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    return input_df

def predict_student_grade(input_data, classifier, scaler):
    """Make predictions using the loaded Random Forest model"""
    try:
        # Preprocess the input data (encoding and scaling)
        input_df = preprocess_input(input_data)
        
        # Scale the input data if scaler is available
        if scaler:
            input_df = scaler.transform(input_df)
        
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
    # Set page title and favicon
    st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓")
    
    # Load model and scaler
    classifier, scaler = load_models()

    # Title
    st.title("🎓 Student Grade Predictor")
    st.write("Predict student performance using a machine learning model")

    # Sidebar for input
    st.sidebar.header("Input Student Information")
    
    # Input fields (only the 10 selected features)
    input_data = {}
    input_data['studytime'] = st.sidebar.slider("Study Time (1-4)", min_value=1, max_value=4, value=2)
    input_data['absences'] = st.sidebar.number_input("Number of Absences (0-93)", min_value=0, max_value=93, value=0)
    input_data['G1'] = st.sidebar.number_input("G1 Grade (0-20)", min_value=0, max_value=20, value=10)
    input_data['G2'] = st.sidebar.number_input("G2 Grade (0-20)", min_value=0, max_value=20, value=10)
    input_data['age'] = st.sidebar.number_input("Age (15-22)", min_value=15, max_value=22, value=18)
    input_data['famsize'] = st.sidebar.selectbox("Family Size", ["LE3", "GT3"])
    input_data['traveltime'] = st.sidebar.slider("Travel Time (1-4)", min_value=1, max_value=4, value=2)
    input_data['failures'] = st.sidebar.number_input("Number of Failures (0-4)", min_value=0, max_value=4, value=0)
    input_data['schoolsup'] = st.sidebar.selectbox("School Support", ["No", "Yes"])
    input_data['higher'] = st.sidebar.selectbox("Higher Education Desire", ["No", "Yes"])

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
