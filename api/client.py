import requests
import streamlit as st

def  get_openai_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke", 
     json={'input': {'topic': input_text}})

    if not response.ok:
        return f"Server error {response.status_code}: {response.text}"
    return response.json()['output']['content']

def get_ollama_response(input_text):
        response = requests.post("http://localhost:8000/poem/invoke",
         json={'input': {'topic': input_text}})

        if not response.ok:
            return f"Server error {response.status_code}: {response.text}"
        return response.json()['output']
    
    
## streamlit app
st.title("LangChain API Client with llama")
input_text = st.text_input("write an essay on :")
input_text2 = st.text_input("write a poem on :") 

if input_text:
   st.write(get_openai_response(input_text))
   
if input_text2:
   st.write(get_ollama_response(input_text2))