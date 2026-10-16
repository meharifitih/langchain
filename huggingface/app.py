import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()

 ##load the groq api
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("Chat Groq with demo")

llm=ChatGroq(groq_api_key=groq_api_key, 
             model_name="openai/gpt-oss-120b")

prompt=ChatPromptTemplate.from_template(
     """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        </context>
        Questions:{input}
        """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        
        st.session_state.embedding=OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader=PyPDFDirectoryLoader("./us_census") ##data ingestion 
        st.session_state.docs = st.session_state.loader.load()  ## data loading 
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ##chunk creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents( st.session_state.docs[:20]) ##splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding) ## vector store with ollama
    
    

prompt1= st.text_input("Enter you question from the document")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("vector stor db is ready")
    

import time

    
if prompt1:
    
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    start= time.process_time()
    response= retriever_chain.invoke({'input':prompt1})
    print("Response time: ",time.process_time()-start)
    st.write(response['answer'])
    
    ## with streamlit expander
    with st.expander("Document similarly Search"):
        ## find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------")
    