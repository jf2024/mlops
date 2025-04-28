import streamlit as st
import joblib
import pandas as pd

st.title("Happiness Score Prediction")

# Load your model
def load_artifacts():
    model_pipeline = joblib.load("Ridge_best_model.joblib")
    return model_pipeline

model_pipeline = load_artifacts()

# Inputs
crime_rate = st.number_input("Crime Rate", value=0.0)
unemployment_rate = st.number_input("Unemployment Rate", value=0.0)
work_life_balance = st.number_input("Work-Life Balance", value=0.0)
freedom = st.number_input("Freedom", value=0.0)

# Prediction button
if st.button("Predict Happiness Score"):
    input_features = pd.DataFrame([{
        'Crime_Rate': crime_rate,
        'Unemployment_Rate': unemployment_rate,
        'Work_Life_Balance': work_life_balance,
        'Freedom': freedom
    }])
    
    # FORCE correct column order
    expected_order = model_pipeline.feature_names_in_
    input_features = input_features[expected_order]
    
    prediction = model_pipeline.predict(input_features)
    st.metric("Predicted Happiness Score", round(prediction[0], 2))

# Batch predictions for uploaded files
st.header("Predict for a Batch of Countries")

batches = st.file_uploader("Upload CSV File", type='csv')

if batches is not None:
    dataframe = pd.read_csv(batches)

    # FORCE correct columns in correct order
    expected_order = model_pipeline.feature_names_in_
    input_features = dataframe[expected_order]

    batch_predictions = model_pipeline.predict(input_features)
    dataframe["Predicted Happiness Score"] = batch_predictions

    st.write(dataframe)

    st.download_button(
        'Download Predictions', 
        data=dataframe.to_csv(index=False).encode('utf-8'), 
        file_name='happiness_predictions.csv', 
        mime='text/csv'
    )


