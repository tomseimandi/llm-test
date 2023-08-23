import streamlit as st
from chroma_s3 import ChromaS3
from db_utils import embeddings
from utils import build_retrieval_qa_filter, set_qa_prompt
from constants import (
    S3_BUCKET_NAME,
    S3_CHROMA_PREFIX,
    CHROMA_LOCAL_PATH
)
from llm import llm


st.title('Document QA prototype - Comptes sociaux')


@st.cache_resource
def get_vectorstore():
    vectordb = ChromaS3.from_s3(
        S3_BUCKET_NAME,
        S3_CHROMA_PREFIX,
        CHROMA_LOCAL_PATH,
        embedding_function=embeddings
    )
    print(f"Vectorstore loaded with {vectordb._collection.count()} documents.")
    return vectordb


def generate_response(dbqa, input_text):
    response = dbqa({'query': input_text})
    st.info(response['result'])


# Loading ChromaDB vectorstore
# TODO: adapt for FAISS
vectordb = get_vectorstore()
qa_prompt = set_qa_prompt()


with st.form('my_form'):
    input_text = st.text_area(
        'Entrez une question :',
        'Quelle société détient SAJARLE SAS ?')

    # Siren et année concernés par la recherche
    company_id = st.text_area('SIREN concerné :', '877913996')
    year = st.text_area('Année :', '2019')
    source = f"{company_id}_{year}.pdf"

    dbqa = build_retrieval_qa_filter(llm, qa_prompt, vectordb, source)

    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(dbqa, input_text)
