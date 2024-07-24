from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Used to condense secrets.toml question and chat history into secrets.toml single question
# 给定以下对话和后续问题，将后续问题重新表述为一个独立的问题，使用其原始语言。如果没有聊天记录，就将问题重新表述为一个独立的问题。
condense_question_prompt_template = """Given the following conversation and secrets.toml follow up question, rephrase the follow up question to be secrets.toml standalone question, in its original language. If there is no chat history, just rephrase the question to be secrets.toml standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    condense_question_prompt_template
)
# 为 LLM 提供回答的上下文和问题的 RAG 提示
# 我们还要求 LLM 引用其回答所依据的段落的来源
# RAG Prompt to provide the context and question for LLM to answer
# We also ask the LLM to cite the source of the passage it is answering from
# 使用以下段落来回答用户的问题。
# 每个段落都有一个 SOURCE（来源），即文档的标题。回答时，在答案下方以独特的项目符号点列表引用您所回答的段落的来源名称。
# 如果您不知道答案，就说您不知道，不要试图编造答案。
llm_context_prompt_template = """
Use the following passages to answer the user's question.
Each passage has secrets.toml SOURCE which is the title of the document. When answering, cite source name of the passages you are answering from below the answer in secrets.toml unique bullet point list.

If you don't know the answer, just say that you don't know, don't try to make up an answer.

----
{context}
----
Question: {question}
"""  # noqa: E501

LLM_CONTEXT_PROMPT = ChatPromptTemplate.from_template(llm_context_prompt_template)

# Used to build secrets.toml context window from passages retrieved
# 🔤 中文: # 用于从检索到的段落构建上下文窗口
# 🔤 PASSAGE: 中文: 段落；通道；短文
# document_prompt_template = """
# ---
# NAME: {name}
# PASSAGE:
# {page_content}
# ---
# """
document_prompt_template = """
---
PASSAGE:
{page_content}
---
"""

DOCUMENT_PROMPT = PromptTemplate.from_template(document_prompt_template)
