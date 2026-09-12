import os
from dotenv import load_dotenv
assert load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
from langsmith import Client
import aiohttp
import asyncio
import json
from variables import SERVER_BASE_URL, EVAL_MODEL_ID

# local objects
from llm_utils import universal_chat_model, make_scorer_with_llm
from variables import MODEL_ID, EMBEDDINGS_MODEL_PROVIDER, LANGSMITH_PROJECT

# full procedure of invoke response and evaluating with LLM as a judge
async def main():
    print(f"LLM model ID:\t\t\t{MODEL_ID}")
    print(f"Embeddings model provider:\t{EMBEDDINGS_MODEL_PROVIDER}")
    print(f"Langsmith Project Name:\t\t{LANGSMITH_PROJECT}")
    confirm = input("Press Enter to confirm:").strip()
    if confirm != "":
        print("Update model IDs and restart the app.")
        exit(0)

    eval_dataset_name = input("Select eval dataset. One of: \"myegpt-22Dec25\" (default), \"test\", or \"test-hard\":") or "myegpt-22Dec25"
    splits = input("Enter split. One of \"base\" (default), \"easy\", \"medium\", \"hard\"):") or "base"
        
    OUTPUT_JSON = f"../responses/examination/{eval_dataset_name}/{LANGSMITH_PROJECT}.json"
    if not os.path.exists(os.path.dirname(OUTPUT_JSON)):
        os.makedirs(os.path.dirname(OUTPUT_JSON))

    # Define the input and reference output pairs that you'll use to evaluate your app
    client = Client()

    eval_llm = universal_chat_model(EVAL_MODEL_ID)
    scorer = make_scorer_with_llm(eval_llm)

    timeout = aiohttp.ClientTimeout(total=600)  # total timeout of 600 seconds
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False),timeout=timeout) as session:
        try:
            results = []

            async def target(inputs: dict) -> dict:
                async with session.post(
                    os.path.join(SERVER_BASE_URL, 'api', 'ask'),
                    headers={
                        "Content-Type": "application/json",
                    },
                    json={"user_input": str(inputs)},
                ) as response_ask:
                    chunks = []
                    async for chunk in response_ask.content.iter_chunked(4096):
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
                experiment_prefix=LANGSMITH_PROJECT,
                metadata={
                    'app_llm': MODEL_ID,
                    'eval_llm': EVAL_MODEL_ID,
                }
            )
            with open(OUTPUT_JSON, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    asyncio.run(main())