import os
import chainlit as cl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.embeddings.cloudflare_workersai import CloudflareWorkersAIEmbeddings


@cl.on_chat_start
async def on_chat_start():
    load_dotenv(override=True)
    qw_llm_openai = ChatOpenAI(
        openai_api_base=os.getenv('DASHSCOPE_API_BASE'),
        openai_api_key=os.getenv('DASHSCOPE_API_KEY'),
        model_name="qwen2-1.5b-instruct",
        temperature=0,
        streaming=True,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | qw_llm_openai | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
