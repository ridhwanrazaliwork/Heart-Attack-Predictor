import os
import warnings
import streamlit as st
import pandas as pd
from PIL import Image
warnings.filterwarnings('ignore')

DATA = os.path.join(os.path.normpath(os.getcwd() + os.sep),'Data', 'Test.csv')
IMAGE_PATH = os.path.join(os.path.normpath(os.getcwd() + os.sep), 'src', 'class_report.png')
df = pd.read_csv(DATA)
image = Image.open(IMAGE_PATH)

st.sidebar.header('Table of Content')
st.sidebar.markdown('''
- [What is a heart attack?](#what-is-a-heart-attack)
- [What are the symptoms of heart attack?](#what-are-the-symptoms-of-heart-attack)
- [What can I do to recover after a heart attack?](#what-can-i-do-to-recover-after-a-heart-attack)
- [Example of Test Data](#example-of-test-data)
''', unsafe_allow_html=True)

st.header('Heart Attack Symptoms, Risk, and Recovery')
st.subheader('What is a heart attack?')

'''A heart attack, also called a myocardial infarction, happens when a part of the heart muscle doesn’t get enough blood.

The more time that passes without treatment to restore blood flow, the greater the damage to the heart muscle.

Coronary artery disease (CAD) is the main cause of heart attack. A less common cause is a severe spasm, or sudden contraction, of a coronary artery that can stop blood flow to the heart muscle.')'''

st.subheader('What are the symptoms of heart attack?')

'''The major symptoms of a heart attack are

- Chest pain or discomfort. Most heart attacks involve discomfort in the center or left side of the chest that lasts for more than a few minutes or that goes away and comes back. The discomfort can feel like uncomfortable pressure, squeezing, fullness, or pain.
- Feeling weak, light-headed, or faint. You may also break out into a cold sweat.
- Pain or discomfort in the jaw, neck, or back.
- Pain or discomfort in one or both arms or shoulders.
- Shortness of breath. This often comes along with chest discomfort, but shortness of breath also can happen before chest discomfort. '''

st.subheader('What can I do to recover after a heart attack?')

''' If you’ve had a heart attack, your heart may be damaged. This could affect your heart’s rhythm and its ability to pump blood to the rest of the body. You may also be at risk for another heart attack or conditions such as stroke, kidney disorders, and peripheral arterial disease (PAD).

You can lower your chances of having future health problems following a heart attack with these steps:

- Physical activity—Talk with your health care team about the things you do each day in your life and work. Your doctor may want you to limit work, travel, or sexual activity for some time after a heart attack.
- Lifestyle changes—Eating a healthier diet, increasing physical activity, quitting smoking, and managing stress—in addition to taking prescribed medicines—can help improve your heart health and quality of life. Ask your health care team about attending a program called cardiac rehabilitation to help you make these lifestyle changes.
- Cardiac rehabilitation—Cardiac rehabilitation is an important program for anyone recovering from a heart attack, heart failure, or other heart problem that required surgery or medical care.'''

st.subheader('Example of test data')
st.dataframe(df)

st.subheader('Dataset Link')
st.write(' #### [Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)',unsafe_allow_html=True)

st.subheader('Model Performance')


st.image(image, caption='Classification report')