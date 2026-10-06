import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv(find_dotenv(), override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(
    model="gpt-5.5",
    api_key=OPENAI_API_KEY
)


prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an Agile Coach. "
        "Answer questions related to Agile processes."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


chain = prompt_template | llm


if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(session_id):

    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()

    return st.session_state.store[session_id]


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


st.title("Agile Guide")


user_input = st.text_input("Enter your question:")


if user_input:

    response = chain_with_history.invoke(
        {"input": user_input},
        config={
            "configurable": {
                "session_id": "abc123"
            }
        }
    )

    st.write(response.content)


st.write("HISTORY")

history = get_session_history("abc123")

for message in history.messages:
    st.write(message.content)