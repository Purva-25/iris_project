import pandas as pd
import numpy as np 
import pickle
import streamlit as st 

# Set page configuration
st.set_page_config(
    page_title='Iris Project Purva', 
    page_icon="🌸", 
    layout='wide', 
    initial_sidebar_state='expanded'
)

# Add a title with some styling
st.title('🌼 Iris Flower Identification 🌼')
st.markdown("## Predict the species of an Iris flower based on its features")

# Create input columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    sep_len = st.number_input('🌿 Sepal Length (cm):', min_value=0.00, step=0.01)
    sep_wid = st.number_input('🌿 Sepal Width (cm):', min_value=0.00, step=0.01)

with col2:
    pet_len = st.number_input('🌸 Petal Length (cm):', min_value=0.00, step=0.01)
    pet_wid = st.number_input('🌸 Petal Width (cm):', min_value=0.00, step=0.01)

# Add a submit button with custom styling
submit = st.button('🔍 Predict')

# Add subheader for the prediction result
st.subheader('🔮 Predictions:')

# Create a function to predict species
def predict_species(scaler_path, model_path):
    with open(scaler_path, 'rb') as file1:
        scaler = pickle.load(file1)
    with open(model_path, 'rb') as file2:
        model = pickle.load(file2)
    
    dct = {
        'SepalLengthCm': [sep_len],
        'SepalWidthCm': [sep_wid],
        'PetalLengthCm': [pet_len],
        'PetalWidthCm': [pet_wid]
    }
    xnew = pd.DataFrame(dct)
    xnew_pre = scaler.transform(xnew)
    pred = model.predict(xnew_pre)
    probs = model.predict_proba(xnew_pre)
    max_prob = np.max(probs)
    
    return pred, max_prob

# Show the result in Streamlit
if submit:
    scaler_path = 'notebook/scaler.pkl'
    model_path = 'notebook/model.pkl'
    pred, max_prob = predict_species(scaler_path, model_path)
    
    st.write('')  
    st.subheader(f'🌟 Predicted Species: **{pred[0]}**')
    st.subheader(f'📊 Probability of Prediction: **{max_prob:.4f}**')
    
    # Add a progress bar for the probability
    st.progress(max_prob)
