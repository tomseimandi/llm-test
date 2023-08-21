import streamlit as st
from utils import setup_dbqa


st.title('🦜🔗 Quickstart App')


def generate_response(input_text):
    dbqa = setup_dbqa()
    response = dbqa({'query': input_text})
    st.info(response['result'])


with st.form('my_form'):
    input_text = st.text_area('Enter text:', 'What is the minimum guarantee payable by Adidas?')

    # Siren et année concernés par la recherche
    company_id = st.text_area('Siren:', '120027016')
    year = st.text_area('Année:', '2021')

    submitted = st.form_submit_button('Submit')

    if submitted:
        generate_response(input_text)
