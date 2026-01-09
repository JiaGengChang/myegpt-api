import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uuid

# src modules
from agent import send_init_prompt, query_agent
from models import Query

print(f"LLM model ID:\t\t\t{os.environ.get('MODEL_ID')}")
print(f"Embeddings model provider:\t{os.environ.get('EMBEDDINGS_MODEL_PROVIDER')}")
print(f"Langsmith Project Name:\t\t{os.environ.get('LANGSMITH_PROJECT')}")
confirm = input("Press Enter to confirm:").strip()
if confirm != "":
    print("Update model IDs and restart the app.")
    exit(0)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET","POST"],
    allow_headers=["Content-Type"],
)

app_dir = os.path.dirname(os.path.abspath(__file__))
graph_folder = os.path.join(app_dir, 'graph')
os.makedirs(graph_folder, exist_ok=True)
static_dir = os.path.join(app_dir, 'static')
scripts_dir = os.path.join(app_dir, 'static', 'scripts')
templates_dir = os.path.join(app_dir, 'templates')
result_folder = os.path.join(app_dir, 'result')
os.makedirs(result_folder, exist_ok=True)

app.mount("/result", StaticFiles(directory=result_folder), name="result") # serve csv files
app.mount("/graph", StaticFiles(directory=graph_folder), name="graph") # serve plotted graphs

# Route(s)
# triggered by submission of query
@app.post("/api/ask")
async def ask(query: Query):
    app.state.username = uuid.uuid4().hex
    app.state.model_id = os.environ.get("MODEL_ID")
    app.state.embeddings_model_id = os.environ.get("EMBEDDINGS_MODEL_ID")
    await send_init_prompt(app)

    def generate_response():
        for chunk in query_agent(query.user_input):
            yield chunk
    return StreamingResponse(generate_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    assert load_dotenv(os.path.join(os.path.dirname(__file__),'.env'))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
    )