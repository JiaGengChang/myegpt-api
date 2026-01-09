import os
import json
from fastapi import FastAPI
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
import matplotlib
matplotlib.use('Agg') # non-interactive backend

from tools import document_search_tool, convert_gene_tool, gene_metadata_tool, gene_level_copy_number_tool, cox_regression_base_data_tool, langchain_query_sql_tool, python_repl_tool, python_execute_sql_query_tool, display_plot_tool, generate_graph_filepath_tool
from llm_utils import universal_chat_model

# Create a system message for the agent
# dynamic variables will be filled in at the start of each session
# removed db description
def create_system_message() -> str:
    db_uri = os.environ.get("COMMPASS_DB_URI")
    db = SQLDatabase.from_uri(db_uri)
    with open(f'{os.path.dirname(__file__)}/prompt.txt', 'r') as f:
        latent_system_message = f.read()
    system_message = latent_system_message.format(
        dialect=db.dialect,
        commpass_db_uri=db_uri
    )
    return [HumanMessage(content='Hello, MyeGPT!'),
            SystemMessage(content=system_message)]

async def send_init_prompt(app:FastAPI):
    global graph
    global config_ask
    config_init = {"thread_id": app.state.username, "recursion_limit": 5} # init configuration
    config_ask = {"thread_id": app.state.username, "recursion_limit": 50} # ask configuration

    #  initialize the chat model
    llm = universal_chat_model(os.environ.get("MODEL_ID"))
    
    graph = create_react_agent(
        model=llm,
        tools=[document_search_tool, convert_gene_tool, gene_metadata_tool, gene_level_copy_number_tool, cox_regression_base_data_tool, langchain_query_sql_tool, python_repl_tool, python_execute_sql_query_tool, generate_graph_filepath_tool, display_plot_tool],
        checkpointer=InMemorySaver(),
    )
    system_message = create_system_message()
    try:
        init_response = await graph.ainvoke({"messages" : system_message}, config_init)
        # Store the init response for injection into HTML
        app.state.init_response = init_response["messages"][-1].content
    except Exception as initialization_error:
        # likely input length exceeded
        app.state.init_response = f"Error during initialization: {initialization_error}"
    

def query_agent(user_input: str):
    global graph
    global config_ask
    user_message = HumanMessage(content=user_input)
    
    for step in graph.stream({"messages": [user_message]}, config_ask, stream_mode="updates"):
        pretty = json.dumps(step, indent=2, ensure_ascii=False, default=str)
        print(pretty)
        yield(pretty)
