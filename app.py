import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler

# using arxiv and wiki tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search_tool = DuckDuckGoSearchRun()

# using grow for llm
groq_api = os.getenv('GROQ_API_KEY')

# using streamlit
st.title("Search Agent using Opensource models")

# creating a new session
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Search agent using arxiv,wiki,duckduckgo tools"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder='ask any question like what is machine learning'): # to display chat input widget
    st.session_state.messages.append({"role":"user","content":prompt}) # appennding the user give prompt
    st.chat_message("user").write(prompt)
    llm = ChatGroq(api_key=groq_api,model='Llama3-8b-8192',streaming=True)
    tools = [arxiv_tool,wiki_tool,search_tool]
    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
