import os
from dotenv import load_dotenv
assert load_dotenv(os.path.join(os.path.dirname(__file__), '.env_eval'))
from langsmith import Client
import aiohttp
import aiofiles
import asyncio
import json
from tqdm import tqdm

client = Client()

# Pure HTTP client
async def main():
    print(f"LLM model ID:\t\t\t{os.environ.get('MODEL_ID')}")
    print(f"Embeddings model provider:\t{os.environ.get('EMBEDDINGS_MODEL_PROVIDER')}")
    print(f"Langsmith Project Name:\t\t{os.environ.get('LANGSMITH_PROJECT')}")
    confirm = input("Press Enter to confirm:").strip()
    if confirm != "":
        print("Update model IDs and restart the app.")
        return
    EVAL_DATASET_NAME = input("Select eval dataset: \"test\", \"test-hard\", or \"myegpt-22Dec25\" (default):") or "myegpt-22Dec25"
    EVAL_SPLITS = input(f"Enter eval split(s): \"easy\", \"medium\", \"difficult\", or \"base\" (default):") or "base"
    
    OUTPUT_JSON = f"../responses/microdocs/{EVAL_DATASET_NAME}/{os.environ.get('LANGSMITH_PROJECT')}.json"
    print(f"Output JSON file: {OUTPUT_JSON}")
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    print("Starting evaluation...")

    timeout = aiohttp.ClientTimeout(total=600)  # total timeout of 600 seconds
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False), timeout=timeout) as session:
        results = []
        try:
            examples = client.list_examples(dataset_name=EVAL_DATASET_NAME, splits=EVAL_SPLITS)
            for example in tqdm(examples):
                inputs = example.inputs
                async with session.post(
                    os.path.join(os.environ.get("SERVER_BASE_URL"), 'api', 'ask'),
                    headers={
                        "Content-Type": "application/json",
                    },
                    json={"user_input": str(inputs)},
                ) as response:
                    chunks = []
                    async for chunk in response.content.iter_chunked(1024):
                        if not chunk:
                            continue
                        chunks.append(chunk.decode("utf-8", errors="strict"))
                    answer = "".join(chunks)
                    output = {"answer": answer}
                    result = {
                        "input": inputs,
                        "output": output
                    }
                    results.append(result)
                try:
                    async with aiofiles.open(OUTPUT_JSON, "w") as f:
                        await f.write(json.dumps(results, indent=2))
                except Exception as file_e:
                    print(f"Error writing to {OUTPUT_JSON}: {file_e}")

        except Exception as e:
            print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
