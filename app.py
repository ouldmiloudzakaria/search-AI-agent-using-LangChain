import os
import certifi
from dotenv import load_dotenv

import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor


# ============================================================
# CONFIGURATION
# ============================================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="LangChain AI Agent",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 LangChain AI Agent")

st.write(
    "Ask me anything. I can search the web using Tavily "
    "when necessary."
)


# ============================================================
# INITIALIZE AGENT
# ============================================================

@st.cache_resource
def create_agent():

    # Tavily search tool
    search_tool = TavilySearchResults(
        max_results=3
    )

    tools = [search_tool]

    # LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        temperature=0,
        model="qwen/qwen3.8-27b",
    )

    # ReAct prompt
    prompt = hub.pull("hwchase17/react")

    # Create agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    return agent_executor


agent_executor = create_agent()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Ask something..."
)


if user_input:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent response
    with st.chat_message("assistant"):

        with st.spinner("🤔 Thinking..."):

            try:

                response = agent_executor.invoke(
                    {
                        "input": user_input
                    }
                )

                answer = response["output"]

                st.markdown(answer)

                # Save response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    f"❌ Error:\n\n{str(e)}"
                )