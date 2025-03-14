import streamlit as st

st.set_page_config(page_title="CKD Prediction", layout="wide")

st.title("Chronic Kidney Disease Prediction Dashboard")

st.sidebar.title("Navigation")

st.image(f"kidney.jpg")

st.markdown("""
Welcome to the **CKD Prediction App**!  
Use the sidebar to explore **data insights** or make a **prediction**.
""")

