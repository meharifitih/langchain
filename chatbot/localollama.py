from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

## langsmith tracing
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant. Please respond to the user queries"),
        ("user","Question: {question}"),
    ]
)

## streamlit app
st.title("LangChain Chatbot demo with smollm2")
input_text = st.text_input("Search the topic you want ")

## Ollama chat model
llm = OllamaLLM(model="smollm2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))