import os
from dotenv import load_dotenv
assert load_dotenv(os.path.join(os.path.dirname(__file__), '.env_eval'))
from langsmith import Client
import aiohttp
import asyncio
import json
import re
# local objects
from llm_utils import universal_chat_model, make_scorer_with_llm

# full procedure of invoke response and evaluating with LLM as a judge
async def main():
    eval_dataset_name = os.environ.get("EVAL_DATASET_NAME")
    if eval_dataset_name:
        print(f"Using eval dataset: {eval_dataset_name}")
    else:
        eval_dataset_name = input("Select eval dataset. One of: \"myegpt-22Dec25\" (default), \"test\", or \"test-hard\":") or "myegpt-22Dec25"
    
    splits = os.environ.get("EVAL_SPLIT")
    if splits:
        splits = re.split(r'\s*,\s*', splits)
        print(f"Using eval dataset splits: {splits}")
    else:
        splits = input("Enter split. One of \"base\" (default), \"easy\", \"medium\", \"hard\"):") or "base"
        
    OUTPUT_JSON = f"../responses/microdocs/{eval_dataset_name}/{os.environ.get('LANGSMITH_PROJECT')}.json"

    # Define the input and reference output pairs that you'll use to evaluate your app
    client = Client()

    eval_llm = universal_chat_model(os.environ.get("EVAL_MODEL_ID"))
    scorer = make_scorer_with_llm(eval_llm)

    timeout = aiohttp.ClientTimeout(total=600)  # total timeout of 600 seconds
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False),timeout=timeout) as session:
        try:
            results = []

            async def target(inputs: dict) -> dict:
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
                        chunks.append(chunk.decode("utf-8", errors="ignore"))

                    answer = "".join(chunks)
                    output = {"answer": answer}
                    results.append({
                        "input": inputs,
                        "output": output
                    })
                    return output

            await client.aevaluate(
                target,
                data=client.list_examples(dataset_name=eval_dataset_name, splits=splits),
                evaluators=[scorer],
                max_concurrency=0,
                num_repetitions=1,
                experiment_prefix=os.environ.get("LANGSMITH_PROJECT"),
                metadata={
                    'app_llm': os.environ.get("MODEL_ID"),
                    'eval_llm': os.environ.get("EVAL_MODEL_ID"),
                }
            )
            with open(f"../responses/microdocs/{eval_dataset_name}/{os.environ.get('LANGSMITH_PROJECT')}.json", "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    asyncio.run(main())