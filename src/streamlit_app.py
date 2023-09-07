import streamlit as st
from chroma_s3 import ChromaS3
from db_utils import embeddings
from utils import build_retrieval_qa_filter, set_qa_prompt
from constants import (
    S3_BUCKET_NAME,
    S3_CHROMA_PREFIX,
    CHROMA_LOCAL_PATH
)
from llm import get_llm
from pathlib import Path


st.title('Document QA prototype')


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


@st.cache_resource
def get_available_documents(_vectordb):
    # TODO: not the right way to do that, to think about
    sources = _vectordb.get(include=["metadatas"]).get("metadatas")
    sources = [tuple(Path(source.get("source")).stem.split("_")) for source in sources]
    available_documents = list(set(sources))
    return available_documents


def generate_response(dbqa, input_text):
    response = dbqa({'query': input_text})
    st.info(
        body=response['result']
    )


# Loading ChromaDB vectorstore
# TODO: adapt for FAISS
vectordb = get_vectorstore()
available_documents = get_available_documents(vectordb)
qa_prompt = set_qa_prompt()
llm = get_llm()


with st.form('my_form'):
    # Siren et année concernés par la recherche
    company_id = st.text_area(
        label='Numéro Siren :office:',
        value='877913996',
        max_chars=9,
        help="Indiquez le numéro Siren d'une entreprise.")
    year = st.text_area(
        label='Année :stopwatch:',
        value='2019',
        max_chars=4,
        help="Indiquez une année.")

    # Question
    input_text = st.text_area(
        label='Question :question:',
        value='Quelle société détient SAJARLE SAS ?',
        help="Posez une question à l'assistant!")

    # Source to filter metadata
    source = f"{company_id}_{year}.pdf"
    dbqa = build_retrieval_qa_filter(llm, qa_prompt, vectordb, source)

    submitted = st.form_submit_button('Submit')
    if submitted:
        if (company_id, year) not in available_documents:
            st.info(
                body=f"Pas de document pour le SIREN {company_id} en {year}."
            )
        else:
            generate_response(dbqa, input_text)
