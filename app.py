import streamlit as st

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import SystemMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

from tools.sql import initialize_connection, list_tables, run_query_tool, describe_tables_tool

def main():
    load_dotenv(find_dotenv())

    chat= ChatOpenAI(streaming= True, temperature=0)

    # Streamlit UI for program name and file upload
    st.set_page_config(page_title="Chat with SQLite Database", page_icon="🤖")
    st.header('Chat with SQLite Database')
    
    data_file= st.file_uploader("Upload the sqlite file", type='sqlite')

    if data_file is not None:
        with open(data_file.name, 'wb') as f:
            f.write(data_file.getbuffer()) 
        initialize_connection(data_file)

        tables= list_tables()
        print(tables)

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
                verbose= True,
                tools= tools,
                memory= memory
            )
            
            chain= agent_executor

            return chain
       
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, ask me anything about the sqlite file you just uploaded.")
            ]

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
        
        if "user_input" not in st.session_state:
            st.session_state.user_input= ""

        # User input
        user_input = st.chat_input('Type your message here')

        if user_input is not None and user_input != "":
            agent_executor= agent_executor_fn()
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            
            with st.chat_message("Human"):
                st.markdown(user_input)
            
            with st.chat_message("AI"):
                ai_response= agent_executor.invoke({'input':user_input})
                st.markdown(ai_response['output'])

            st.session_state.chat_history.append(AIMessage(content= ai_response['output']))
            

        # # Display chat history
        # chat_container= st.container()
        # with chat_container:
        #     for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
        #         st.markdown(f"**User:** {user_msg}")
        #         st.markdown(f"**AI:** {ai_msg}")

if __name__ == '__main__':
    main()

