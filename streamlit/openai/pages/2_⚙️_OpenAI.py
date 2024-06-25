import streamlit as st

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""

if "OPENAI_API_BASE" not in st.session_state:
    st.session_state["OPENAI_API_BASE"] = ""

st.set_page_config(page_title="OpenAI Settings", layout="wide")

st.title("OpenAI Settings")

openai_api_key = st.text_input("API Key", value=st.session_state["OPENAI_API_KEY"], max_chars=None, key=None,
                               type='password')
openai_api_base = st.text_input("BASE URL", value=st.session_state["OPENAI_API_BASE"], max_chars=None, key=None,
                                type='default')

saved = st.button("Save")

if saved:
    st.session_state["OPENAI_API_KEY"] = openai_api_key
    st.session_state["OPENAI_API_BASE"] = openai_api_base
