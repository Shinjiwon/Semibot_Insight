import streamlit as st
from pathlib import Path
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentType, initialize_agent
from langchain.schema import SystemMessage
from langchain.agents import Tool
from langchain.prompts import PromptTemplate,ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

import os

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

os.environ["OPENAI_API_KEY"]    = st.secrets.OPENAI_API_KEY
 
os.environ["GOOGLE_API_KEY"]    = st.secrets.GOOGLE_API_KEY
os.environ["GOOGLE_CSE_ID"]     = st.secrets.GOOGLE_CSE_ID

os.environ["DB_USER"]   = st.secrets.DB_USER
os.environ["DB_PW"]     = st.secrets.DB_PW
os.environ["DB_SERVER"] = st.secrets.DB_SERVER
os.environ["DB_NAME"]   = st.secrets.DB_NAME
os.environ["DB_URI"]    = st.secrets.DB_URL

#Datasource
database_user     = os.getenv("DB_USER")
database_password = os.getenv("DB_PW")
database_server   = os.getenv("DB_SERVER")
database_db       = os.getenv("DB_NAME")

#Connection String
import urllib.parse
encoded_password = urllib.parse.quote(database_password)

connection_string = f"mssql+pymssql://{database_user}:{encoded_password}@{database_server}:1433/{database_db}"

#Include tables
include_tables=[ 'TEMP_TB_QMS_DYN_VEND_RESULT',
                'TB_QMS_COMM_FILE', 
                'TB_QMS_DYN_ACTIVITY',
                'TB_QMS_DYN_ATTACH_CONTENTS',
                'TB_QMS_DYN_AUTO_EMAIL',
                'TB_QMS_DYN_AUTO_EMAIL_NBALIVE21',
                'TB_QMS_DYN_DEFECT_STATUS_CR',
                'TB_QMS_DYN_ENV_CHECK',
                'TB_QMS_DYN_EVENT',
                'TB_QMS_DYN_FINAL_INSP',
                'TB_QMS_DYN_FINAL_INSP_DEF',
                'TB_QMS_DYN_FINAL_INSP_MEAS',
                'TB_QMS_DYN_IMP_INSP',
                'TB_QMS_DYN_IMP_INSP_DEF',
                'TB_QMS_DYN_IMP_INSP_MEAS',
                'TB_QMS_DYN_MEETING_MEM',
                'TB_QMS_DYN_REJECT',
                'TB_QMS_DYN_ROUTE_INSP',
                'TB_QMS_DYN_ROUTE_INSP_DEF',
                'TB_QMS_DYN_ROUTE_INSP_MEAS',
                'TB_QMS_DYN_VEND_RESULT',
                'TB_QMS_MAP_BP_ITEM',
                'TB_QMS_MAP_MENU_DIM',
                'TB_QMS_MAP_MENU_MEAS',
                'TB_QMS_MAP_ROLE_MENU_ACTION',
                'TB_QMS_MAP_USER_MENU_DIM',
                'TB_QMS_MAP_USER_MENU_MEAS',
                'TB_QMS_MAP_USER_ROLE',
                'TB_QMS_MAP_VEND_BCD',
                'TB_QMS_MST_ANNUAL_TARGET',
                'TB_QMS_MST_CODE',
                'TB_QMS_MST_COORDI',
                'TB_QMS_MST_DIM',
                'TB_QMS_MST_EMAIL_AUTH',
                'TB_QMS_MST_EVENT_MAPPING',
                'TB_QMS_MST_INSP',
                'TB_QMS_MST_INSP_ROUTE',
                'TB_QMS_MST_ITEM_DRAW_REV_INFO',
                'TB_QMS_MST_ITEM_INSP',
                'TB_QMS_MST_MEAS',
                'TB_QMS_MST_MENU',
                'TB_QMS_MST_MENU_FAVORITES',
                'TB_QMS_MST_NOTICE',
                'TB_QMS_MST_ROLE',
                'TB_QMS_MST_TRANSLATE',
                'TB_QMS_MST_USER',
                'TB_QMS_MST_USER_ATTB',
                'TB_QMS_MST_VEND',
                'TB_QMS_MST_VEND_TAR'
 ]

openai_api_key = os.getenv("OPENAI_API_KEY")

# Check user inputs
if not connection_string:
    st.info("Please enter database URI to connect to your database.")
    st.stop()

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# Setup agent
#llm = OpenAI(openai_api_key=openai_api_key, temperature=0, streaming=True)
llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0,  streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri, include_tables=include_tables)


db = configure_db(connection_string)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

from langchain.prompts import PromptTemplate

custom_suffix = """
You must query using MSSQL.
Be sure to answer in Korean
"""

agent_template = """
  You are an expert MSSQL data analyst.You must query using mssql syntax.
  Be sure to answer in Korean!

  {memory}
  Human: {human_input}
Chatbot:"""

agent_prompt = PromptTemplate(input_variables=["memory", "human_input"],template=agent_template)

agent_memory = ConversationBufferMemory(memory_key="memory",prompt=agent_prompt, return_messages=True)

agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="agent_memory")],
        }
# conversational memory
conversational_memory = ConversationBufferMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=False,
    memory=conversational_memory,
    agent_kwargs=agent_kwargs,
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)