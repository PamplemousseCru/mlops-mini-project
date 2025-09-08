import joblib
import streamlit as st

model = joblib.load("regression.joblib")

size_input = st.number_input("size", min_value=0)
nb_bedrooms_input = st.number_input("nb_bedrooms", min_value=0, step=1)
garden_input = st.number_input("garden", min_value=0, max_value=1, step=1)

result = model.predict([[size_input, nb_bedrooms_input, garden_input]])
st.write(result)
