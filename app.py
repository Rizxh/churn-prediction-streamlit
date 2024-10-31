import pandas as pd
import pickle
import streamlit as st

# Load the pre-trained model
model_loaded = pickle.load(open('final_tuned_model_LightGBM.sav', 'rb'))

# Set up the title of the app
st.write('''
# CHURN CUSTOMER PREDICTOR
''')

# Function to collect user inputs for customer features
def user_input_feature():
    # Input fields in the main page
    st.header("Please input customer's features")
    
    # Numerical inputs
    Tenure = st.number_input('Tenure', min_value=1, max_value=61, value=5, step=1)
    WarehouseToHome = st.number_input('WarehouseToHome', 1, 33, 15, 1)
    NumberOfDeviceRegistered = st.number_input('NumberOfDeviceRegistered', 1, 6, 3, 1)
    DaySinceLastOrder = st.number_input('DaySinceLastOrder', 1, 30, 3, 1)
    CashbackAmount = st.number_input('CashbackAmount', 0, 500, 100, 1)
    NumberOfAddress = st.number_input('NumberOfAddress', 1, 22, 5, 1)

    # Categorical inputs
    MaritalStatus = st.selectbox('MaritalStatus', ('Single', 'Married', 'Divorced'))
    Complain = st.selectbox('Complain', (0, 1))
    PreferedOrderCat = st.selectbox('PreferedOrderCat', ('Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'))
    SatisfactionScore = st.selectbox('SatisfactionScore', (1, 2, 3, 4, 5))

    # Create DataFrame to store inputs
    df = pd.DataFrame({
        'Tenure': [Tenure],
        'WarehouseToHome': [WarehouseToHome],
        'NumberOfDeviceRegistered': [NumberOfDeviceRegistered],
        'MaritalStatus': [MaritalStatus],
        'PreferedOrderCat': [PreferedOrderCat],
        'SatisfactionScore': [SatisfactionScore],
        'NumberOfAddress': [NumberOfAddress],
        'CashbackAmount': [CashbackAmount],
        'Complain': [Complain],
        'DaySinceLastOrder': [DaySinceLastOrder]
    })
    return df

# Collect customer data
df_customer = user_input_feature()
df_customer.index = ['value']

# Make prediction and get probability
kelas = model_loaded.predict(df_customer)
probabilities = model_loaded.predict_proba(df_customer)

# Create two columns on the main page for customer features and prediction result
col1, col2 = st.columns(1)

# Display the customer features in the left column

# Display the prediction result in the right column
with col2:
    st.subheader("Prediction")
    st.write("0 means No")
    st.write("1 means Yes")
    
    if kelas == 1:
        st.write('Result 1: this customer will CHURN')
        st.write(f"Prediction Probability: {probabilities[0][1]:.2f}")
    else:
        st.write('Result 0: this customer will STAY')
        st.write(f"Prediction Probability: {probabilities[0][0]:.2f}")
