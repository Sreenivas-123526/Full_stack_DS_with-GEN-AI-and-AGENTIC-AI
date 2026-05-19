import streamlit as st
import pickle
import numpy as np

#Load the saved model
model=pickle.load(open(r'C:\Users\hp\FullStack_DataScience with Gen AI an d Agentic AI\ML_Linear_regression_model\linear_regression_model.pkl','rb'))

#set the title of the streamlit app
st.title("Salary prediction app")

#Add a brief description
st.write("This app predicts the salary based on years of experience using a simple linear regression model")

#add input widget for user to enter the years of experience
years_experience=st.number_input("Enter the Years of experience",min_value=0.0,max_value=50.0,value=1.0,step=0.5)

#when button is clicked ,make predictions
if st.button("Predict Salary"):
    #Make a prediction using the trained model
    experience_input=np.array([[years_experience]]) #convert the input to 2D array for predictions
    prediction=model.predict(experience_input)

    #display the result
    st.success(f"the predicted salary for {years_experience} years of experience  is : ${prediction[0]:,.2f}")

#Display information about the model
st.write("The model was trained using a dataset of salaries and years of experience.built model by Sreenivas")
