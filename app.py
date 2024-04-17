import streamlit as st
import pandas as pd
import numpy as np  
import joblib
import pickle
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")


# Set favicon with emoji
st.set_page_config(page_title='Employee Performance Evaluation 👨‍💼', page_icon='📊')

# Add emoji to the title
st.title('Employee Performance Evaluation 👨‍💼')
Age = st.number_input('Age',  value=18)
Gender = st.selectbox('Gender', options=['Male','Female'])
EducationBackground = st.selectbox('EducationBackground', options=['Marketing','Life Sciences','Medical','Other','Human Resources','Technical Degree'])
MaritalStatus = st.selectbox('MaritalStatus', options=['Single','Married','Divorced'])
EmpDepartment = st.selectbox('EmpDepartment', options=['Sales','Research & Development','Human Resources']) 
EmpJobRole = st.selectbox('EmpJobRole', options=['Sales Executive','Research Scientist','Laboratory Technician','Manufacturing Director','Healthcare Representative','Manager','Sales Representative','Research Director','Human Resources'])
BusinessTravelFrequency = st.selectbox('BusinessTravelFrequency', options=['Travel_Rarely','Travel_Frequently','Non-Travel'])
DistanceFromHome = st.number_input('DistanceFromHome',  value=18)
EmpEducationLevel = st.number_input('EmpEducationLevel',  value=1)
EmpEnvironmentSatisfaction = st.number_input('EmpEnvironmentSatisfaction',  value=1)
EmpHourlyRate = st.number_input('EmpHourlyRate',  value=1)
EmpJobInvolvement = st.number_input('EmpJobInvolvement',  value=1)
EmpJobLevel = st.number_input('EmpJobLevel',  value=1)
EmpJobSatisfaction = st.number_input('EmpJobSatisfaction',  value=1)
NumCompaniesWorked = st.number_input('NumCompaniesWorked',  value=1)
OverTime = st.selectbox('OverTime', options=['Yes','No'])
EmpLastSalaryHikePercent = st.number_input('EmpLastSalaryHikePercent',  value=1)
EmpRelationshipSatisfaction = st.number_input('EmpRelationshipSatisfaction',  value=1)
TotalWorkExperienceInYears = st.number_input('TotalWorkExperienceInYears',  value=1)
TrainingTimesLastYear = st.number_input('TrainingTimesLastYear',  value=1)
EmpWorkLifeBalance = st.number_input('EmpWorkLifeBalance',  value=1)
ExperienceYearsAtThisCompany = st.number_input('ExperienceYearsAtThisCompany',  value=1)
ExperienceYearsInCurrentRole = st.number_input('ExperienceYearsInCurrentRole',  value=1)
YearsSinceLastPromotion = st.number_input('YearsSinceLastPromotion',  value=1)
YearsWithCurrManager = st.number_input('YearsWithCurrManager',  value=1)
Attrition = st.selectbox('Attrition', options=['Yes','No'])

# Load the label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Encode the object variables
Gender_encoded = label_encoders['Gender'].transform([Gender])[0]
EducationBackground_encoded = label_encoders['EducationBackground'].transform([EducationBackground])[0]
MaritalStatus_encoded = label_encoders['MaritalStatus'].transform([MaritalStatus])[0]
EmpDepartment_encoded = label_encoders['EmpDepartment'].transform([EmpDepartment])[0]
EmpJobRole_encoded = label_encoders['EmpJobRole'].transform([EmpJobRole])[0]
BusinessTravelFrequency_encoded = label_encoders['BusinessTravelFrequency'].transform([BusinessTravelFrequency])[0]
OverTime_encoded = label_encoders['OverTime'].transform([OverTime])[0]
Attrition_encoded = label_encoders['Attrition'].transform([Attrition])[0]

# Use the encoded variables in your code
# ...
if st.button('Predict'):
    # Create a DataFrame with the entered data
    data = pd.DataFrame({
        'Age': [Age],
        'Gender': [Gender_encoded],
        'EducationBackground': [EducationBackground_encoded],
        'MaritalStatus': [MaritalStatus_encoded],
        'EmpDepartment': [EmpDepartment_encoded],
        'EmpJobRole': [EmpJobRole_encoded],
        'BusinessTravelFrequency': [BusinessTravelFrequency_encoded],
        'DistanceFromHome': [DistanceFromHome],
        'EmpEducationLevel': [EmpEducationLevel],
        'EmpEnvironmentSatisfaction': [EmpEnvironmentSatisfaction],
        'EmpHourlyRate': [EmpHourlyRate],
        'EmpJobInvolvement': [EmpJobInvolvement],
        'EmpJobLevel': [EmpJobLevel],
        'EmpJobSatisfaction': [EmpJobSatisfaction],
        'NumCompaniesWorked': [NumCompaniesWorked],
        'OverTime': [OverTime_encoded],
        'EmpLastSalaryHikePercent': [EmpLastSalaryHikePercent],
        'EmpRelationshipSatisfaction': [EmpRelationshipSatisfaction],
        'TotalWorkExperienceInYears': [TotalWorkExperienceInYears],
        'TrainingTimesLastYear': [TrainingTimesLastYear],
        'EmpWorkLifeBalance': [EmpWorkLifeBalance],
        'ExperienceYearsAtThisCompany': [ExperienceYearsAtThisCompany],
        'ExperienceYearsInCurrentRole': [ExperienceYearsInCurrentRole],
        'YearsSinceLastPromotion': [YearsSinceLastPromotion],
        'YearsWithCurrManager': [YearsWithCurrManager],
        'Attrition': [Attrition_encoded]
    })

    # Load the trained model
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = joblib.load(f)

        

        
    # Make predictions
    predictions = model.predict(data)

    # Display the predictions
    
    performance = ""
    if predictions[0] == 2:
        performance = "bad"
        emoji = "😞"
    elif predictions[0] == 3:
        performance = "Normal"
        emoji = "😐"
    elif predictions[0] == 4:
        performance = "good"
        emoji = "😄"

    st.markdown(
        f"<div style='background-color: #f9f2d4; padding: 5px; border-radius: 10px;'>"
        f"<h3 style='color: #6e511f;'>🔮 Employee Performance is : {performance} {emoji}</h3>"
        "</div>",
        unsafe_allow_html=True
    )
