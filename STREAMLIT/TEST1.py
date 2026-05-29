import streamlit as st

st.title('Online Food ordering app')

menu={
    "pizza":250,
    "burger":150,
    "Meal":290
}

if 'cart' not in st.session_state:
    st.session_state.cart={}

for price,item in menu.items():
    col1,col2,col3=st.columns[1,1,1]
    col1=st.write(f'{item}')
