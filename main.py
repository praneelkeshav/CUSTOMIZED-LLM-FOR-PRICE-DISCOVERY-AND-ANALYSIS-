import streamlit as st
from Retrieval import create_vector_db, get_qa_chain


st.title("CUSTOMIZED LLM FOR PRICE DISCOVERY AND ANALYSIS📈💹🪂")
btn = st.button("ENTER YOUR QUERY")

if btn:
    create_vector_db()

question = st.text_input("Question:")

if question:
    chain = get_qa_chain(allow_dangerous_deserialization=True)
    response = chain(question)

    st.header("Answer:")
    st.write(response["result"])

