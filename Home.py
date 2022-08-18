import os
import pickle
import numpy as np
import warnings
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
warnings.filterwarnings('ignore')
# image = Image.open('Logo.png')

# Import trained model
MODEL_PATH = os.path.join(os.getcwd(),'Models', 'best_estimator.pkl')

with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)

DATA = os.path.join(os.getcwd(), 'Data', 'heart.csv')
df = pd.read_csv(DATA)

IMAGE_PATH = os.path.join(os.getcwd(), 'src', 'Logo.png')
image = Image.open(IMAGE_PATH)
st.image(image,use_column_width=True)
"""
# Heart Attack Prediction :heartbeat:

"""

with st.form("Input Your details"):
    st.write("Patient Input")
    age = int(st.number_input('Insert your age',help='Age in years', 
                         min_value=0,max_value=200, 
                         value=21, step=1))
    sex = st.selectbox('Select your gender',
                           options=['Male', 'Female'],
                           index=0)
    cp = st.selectbox('Chest pain type',
                               options=['Typical angina', 'Atypical angina', 
                               'Non-anginal pain', 'Asymptomatic'],
                               index=0)
    rest_bp = float(st.number_input('Resting blood pressure (mm Hg)', 
                                     value=120,min_value=60,max_value=370))
    chol = float(st.number_input('Cholesterol level (mg/dL)', 
                                  value=240,min_value=10,max_value=3170))
    fbs = st.selectbox('Having fasting blood sugar more than 120 mg/dL?',
                                     options=['Yes', 'No'],
                                     index=1)
    rest_ecg = st.selectbox('Resting electrocardiographic results',help='Value 0: normal  '
    'Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)  ' 
    'Value 2: showing probable or definite left ventricular hypertrophy by Estes criteria',
                             options=['Normal', 'Abnormal', 'High-Risk'],
                             index=0)
    thalach = float(st.number_input('Max heart rate', 
                                            value=150,min_value=27, max_value=480))
    exang = st.selectbox('Having exercise-induced angina?',
                                    options=['Yes', 'No'],
                                    index=1)
    old_peak = float(st.number_input('ST depression induced by exercise relative to rest', 
                                      value=1.0))
    slp = st.selectbox('Slope of the peak exercise ST segment',
                            options=['Unsloping', 'Flat', 'Downsloping'],
                            index=2)
    caa = int(st.selectbox('Number of major vessels colored by flourosopy',
                            options=['0', '1', '2', '3'],
                            index=0))
    thall = st.selectbox('Thalassemia',
                              options=['Fixed defect', 'Normal', 'Reversable defect'],
                              index=2)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write('Patient info:')
        d = {'Age': age, 'Gender': sex, 'Chest pain type': cp,
                 'Resting blood pressure': rest_bp,'Cholesterol level': chol,
                 'Fasting blood sugar': fbs,'Resting ECG result': rest_ecg,
                 'Max heart rate': thalach,'Exercise-induced angina': exang,
                 'Oldpeak': old_peak,'SLP': slp,'CAA': caa,
                 'THALL': thall}
        df1 = pd.DataFrame(data=d, index=[1])
        st.dataframe(df1)

        gender_map=1 if sex=='Male' else 0

        if cp == 'Typical angina':
            cp_map = 0
        elif cp == 'Atypical angina':
            cp_map = 1
        elif cp == 'Non-anginal pain':
            cp_map = 2
        elif cp == 'Asymptomatic':
            cp_map = 3


        fbs_map=1 if fbs=='Yes' else 0

        if rest_ecg == 'Normal':
            rest_ecg_map = 0
        elif rest_ecg == 'Abnormal':
            rest_ecg_map = 1
        elif rest_ecg == 'High-Risk':
            rest_ecg_map = 2


        exang_map=1 if exang=='Yes' else 0

        if slp == 'Unsloping':
            slp_map = 0
        elif slp == 'Flat':
            slp_map = 1
        elif slp == 'Downsloping':
            slp_map = 2


        if thall == 'Fixed defect':
            thall_map = 1
        elif thall == 'Normal':
            thall_map = 2
        elif thall == 'Reversable defect':
            thall_map = 3

        
        import numpy as np
        input_row = np.array([age, gender_map, cp_map, rest_bp, chol, 
                              fbs_map,rest_ecg_map, 
                              thalach, exang_map, 
                              old_peak, slp_map,caa, thall_map])

        input_data = input_row.reshape(1, -1)
        y_pred = model.predict(input_data)[0]
        y_list = ['possess a minimal risk of having a heart attack', 'have a greater likelihood of suffering a heart attack']
        y_result = y_list[y_pred]

        probas = model.predict_proba(input_data)[0]
        proba = probas[y_pred]
    
        st.write(f'The patient {y_result}. :smiley:') 
        st.write(f'Probability of having heart attack {proba*100: .2f}\%')


#sidebar
st.sidebar.header('Help & FAQs :question:')
st.sidebar.markdown('Please hover on ? for more information about input')
st.sidebar.markdown('> For more information please click **Background** page above')


