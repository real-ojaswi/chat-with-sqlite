import streamlit as st

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv, find_dotenv

from tools.sql import initialize_connection, list_tables, run_query_tool, describe_tables_tool

def main():
    load_dotenv(find_dotenv())

    chat= ChatOpenAI()

    # Streamlit UI for program name and file upload
    st.header('Chat with SQLite Database')
    
    data_file= st.file_uploader("Upload the sqlite file", type='sqlite')

    if data_file is not None:
        with open(data_file.name, 'wb') as f:
            f.write(data_file.getbuffer()) 
        initialize_connection(data_file)

        tables= list_tables()

        @st.cache_resource
        def agent_executor_fn():
            prompt= ChatPromptTemplate(
                messages=[
                    SystemMessage(
                        "You are an AI that has access to a SQLite Database. \n"
                        f"The database has tables of: {tables}\n"
                        "Do not make any assumptions about what tables exist"
                        "or what columns exist. Instead use the 'describe_tables' function."
                        ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                    
                ]
            )

            memory= ConversationBufferMemory(memory_key="chat_history", return_messages= True)
            tools= [run_query_tool, describe_tables_tool]

            agent= create_openai_functions_agent(
                llm= chat,
                prompt= prompt,
                tools= tools,
            )

            agent_executor= AgentExecutor(
                agent=agent,
                verbose= False,
                tools= tools,
                memory= memory
            )

            return agent_executor
       
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "user_input" not in st.session_state:
            st.session_state.user_input= ""

        # User input
        user_input = st.text_input('Ask questions about the SQL file you just uploaded')

        if user_input or st.button('Send'):
            agent_executor= agent_executor_fn()
            result = agent_executor.invoke({'input': user_input})
            st.session_state.chat_history.append((user_input, result['output']))
            st.session_state.user_input = ""

        # Display chat history
        chat_container= st.container()
        with chat_container:
            for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
                st.markdown(f"**User:** {user_msg}")
                st.markdown(f"**AI:** {ai_msg}")

if __name__ == '__main__':
    main()

