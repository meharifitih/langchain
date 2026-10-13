from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## langsmith tracing
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## prompt template
prompt = ChatPromptTemplate.from_template(
    [
        ("system","you are a helpful assistant. Please respond to the user queries"),
        ("user","Question: {question}"),
    ]
)

## streamlit app
st.title("LangChain Chatbot demo with OpenAI")
input_text = st.input_text("Search the topic you want ")

## oepnai chat model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))