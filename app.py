def preprocess_input(input_data):
    """
    Preprocess input data using only 10 specific features
    
    Args:
        input_data (dict): Input data dictionary
    
    Returns:
        pd.DataFrame: Preprocessed input data
    """
    # List of the 10 features we'll use
    selected_columns = [
        'studytime', 'absences', 'G1', 'G2', 'age', 
        'famsize', 'traveltime', 'failures', 'schoolsup', 'higher'
    ]
    
    # Label Encoder untuk variabel kategorik
    label_encoder = LabelEncoder()
    
    # Variabel kategorik yang perlu di-encode
    categorical_columns = ['famsize', 'schoolsup', 'higher']
    
    # Buat salinan input_data untuk menghindari modifikasi langsung
    processed_data = input_data.copy()
    
    # Encode variabel kategorik
    for col in categorical_columns:
        if col in processed_data:
            # Ubah variabel kategorik menjadi numerik
            processed_data[col] = label_encoder.fit_transform([processed_data[col]])[0]
    
    # Buat DataFrame dengan kolom yang dipilih
    input_df = pd.DataFrame([processed_data])[selected_columns]
    
    return input_df
