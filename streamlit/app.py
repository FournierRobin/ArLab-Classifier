import streamlit as st
import requests

st.title('Sports Articles Classifier')

api_url = "http://api:8000/predict"

text_input = st.text_area("Enter your text here:", height=200)

if st.button("Submit"):
    response = requests.post(api_url, json={"text": text_input})
    
    if response.status_code == 200:
        prediction = response.json()['label']
        if prediction == "Aucun model disponible":
            st.subheader(prediction)
        else:    
            st.subheader(f"Ce texte parle de {prediction}")
    else:
        st.subheader("Error:", response.status_code)

st.header("Liens utiles")

st.markdown(
    """
    <style>
    .boxed {
        background-color: #ffffff;
        border: 2px solid #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <div class="boxed">
        <a href='https://app.clear.ml' target='_blank' rel='noopener noreferrer'>
            <img src='https://repository-images.githubusercontent.com/191126383/7481bc00-4705-11eb-9fed-772d9ed73e28' width='100' />
        </a>
    </div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <div class="boxed">
        <a href='http://localhost:8080' target='_blank' rel='noopener noreferrer'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/d/de/AirflowLogo.png' width='100' />
        </a>
    </div>
    
""", unsafe_allow_html=True)