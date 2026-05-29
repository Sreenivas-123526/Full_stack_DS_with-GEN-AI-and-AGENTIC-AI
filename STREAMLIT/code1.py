#1. Import streamlit
import streamlit as st

#2.add a title to your app
st.title("My first streamlit app created by Sreenivas")

#3.Add some text
st.write("Welcome Gentlemen! and this app will calculate teh square of a number")

#4. create a interactive slider
st.header("select a Number")
number=st.slider("Pick a number",0,100,5) #min,max,default

#5.calculate and display the result
st.subheader("Result is")
squared_number=number*number
st.write(f"the square of {number} is {squared_number}")



#st.title(): this command sets the main title that appears at the top of your web app
#st.write(): A versatile command used to write text,numbers, or even dataframes to the app
#st.header() and st.subheader():These are for creating section headings of different sizes
#st.slider(): This is an interactive widget. it creates a slider that lets the user select a number. The selected value is stored in the number variable
