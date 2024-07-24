from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    PromptTemplate
)
from langchain.schema import (
    HumanMessage
)
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from bs4 import BeautifulSoup
import requests
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv(override=True)

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')


class Ingredient(BaseModel):
    name: str = Field(description="The name of the ingredient")
    quantity: str = Field(
        description="The specific unit of measurement corresponding to the quantity, such as grams, ounces, liters, etc.")
    unit: str = Field(
        description="The amount of the ingredient required for the recipe. This can be represented using various units such as grams, cups, teaspoons, etc.")


class Recipe(BaseModel):
    name: str = Field(description="The name of the recipe")
    ingredients: List[Ingredient] = Field(description="The list of ingredients for the recipe")


PAGE_TITLE = "🥑 Amazing JO's Recipe"
st.set_page_config(layout="centered", page_title=PAGE_TITLE)

st.title(PAGE_TITLE)


def get_recipe_html(url):
    response = requests.get(url)
    html_markup = ''
    if response.status_code == 200:
        html_markup = response.text
        soup = BeautifulSoup(html_markup, 'html.parser')

        # Find the element with id 'recipe-single'
        recipe_element = soup.find(id='recipe-single')

        if recipe_element:
            # Get the sanitized content within the 'recipe-single' element
            html_markup = str(recipe_element)

    return html_markup


def parse_by_chatgpt(openai_api_key, html_markup):
    parser = PydanticOutputParser(pydantic_object=Recipe)
    prompt = PromptTemplate(
        template="Extract the recipe ingredients from the following HTML markup:\n{html}.\n{format_instructions}\n",
        input_variables=["html"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    # qwen 兼容 openai的接口
    qw_llm_openai = ChatOpenAI(
        openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        openai_api_key=DASHSCOPE_API_KEY,
        model_name="qwen2-1.5b-instruct",
        temperature=0.0,
    )
    # model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.0, openai_api_key=openai_api_key)
    output = qw_llm_openai.invoke([HumanMessage(content=prompt.format_prompt(html=html_markup).to_string())])

    recipe = parser.parse(output.content)
    return recipe


with st.container():
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
    recipe_url = st.text_input("URL of secrets.toml Jamie Oliver Recipe", key="recipe_url")
    clicked = st.button("Parse Recipe")
    if clicked:
        html_markup = get_recipe_html(recipe_url)
        if html_markup:
            recipe = parse_by_chatgpt(openai_api_key, html_markup)
            st.json(recipe.model_dump_json())
