import streamlit as st
from utils import build_retrieval_qa_filter, load_vectorstore, set_qa_prompt
from constants import DATABASE_NAME
from llm import llm


st.title('🦜🔗 Quickstart App')


def generate_response(dbqa, input_text):
    response = dbqa({'query': input_text})
    st.info(response['result'])


with st.form('my_form'):
    input_text = st.text_area('Enter text:', 'What is the minimum guarantee payable by Adidas?')

    # Siren et année concernés par la recherche
    company_id = st.text_area('Siren:', '877913996')
    year = st.text_area('Année:', '2019')

    source = f"data/{company_id}_{year}.pdf"
    vectordb = load_vectorstore(DATABASE_NAME)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa_filter(llm, qa_prompt, vectordb, source)

    submitted = st.form_submit_button('Submit')

    if submitted:
        generate_response(dbqa, input_text)
