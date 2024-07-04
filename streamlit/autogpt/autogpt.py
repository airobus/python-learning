from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain  # , SequentialChain
from langchain.chains.sequential import SequentialChain
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
# 基本配置
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

qw_llm_openai = ChatOpenAI(
    openai_api_base=os.getenv('DASHSCOPE_API_BASE'),
    openai_api_key=os.getenv('DASHSCOPE_API_KEY'),
    model_name="qwen2-1.5b-instruct",
    temperature=0,
)

st.title('AutoGPT Wizard')
prompt = st.text_input('Tell me a topic you want to learn its programming language:')

# Prompt templates
language_template = PromptTemplate(
    input_variables=['topic'],
    template='Suggest me a programming language for {topic} and respond in a code block with the language name only'
)

book_recommendation_template = PromptTemplate(
    input_variables=['programming_language'],
    template='''Recommend me a book based on this programming language {programming_language}

    The book name should be in a code block and the book name should be the only text in the code block
    '''
)

# llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
language_chain = LLMChain(llm=qw_llm_openai, prompt=language_template, verbose=True, output_key='programming_language')
# 在新的管道方式中，verbose=True 参数可能不再适用。
# 对于 output_key='programming_language' ，在新的实现方式中可能不需要以这种方式指定输出键。具体取决于您后续如何处理生成的结果
# language_chain = RunnablePassthrough() | language_template | qw_llm_openai

book_recommendation_chain = LLMChain(llm=qw_llm_openai, prompt=book_recommendation_template,
                                     verbose=True, output_key='book_name')

# book_recommendation_chain = RunnablePassthrough() | book_recommendation_template | qw_llm_openai

# auto-gpt 怎么感觉像mult chain 呢 🤔

sequential_chain = SequentialChain(
    chains=[language_chain, book_recommendation_chain],
    input_variables=['topic'],
    output_variables=['programming_language', 'book_name'],
    verbose=True)

if prompt:
    reply = sequential_chain.invoke({'topic': prompt})

    with st.expander("Result"):
        st.info(reply)

    with st.expander("Programming Language"):
        st.info(reply['programming_language'])

    with st.expander("Recommended Book"):
        st.info(reply['book_name'])
