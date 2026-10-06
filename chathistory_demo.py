import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv(find_dotenv(), override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
prompt_template = ChatPromptTemplate.from_messages(
[
    ("system", "You are a Agile Coach.Answer any questions "
               "related to the agile process"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
]
)

chain = prompt_template | llm

history_for_chain = InMemoryChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

print("Agile Guide")

while True:
    question = input("Enter the question:")
    if question == "exit":
        break

    response = (chain_with_history
                .invoke({"input": question},
                        {"configurable":{
                            "session_id":"abc123"
                        }}))
    print(response.content)