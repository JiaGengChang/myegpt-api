import os
from dotenv import load_dotenv
assert load_dotenv(os.path.join(os.path.dirname(__file__), '.env_eval'))
import json
from langsmith import Client
from tqdm import tqdm
# local objects
from llm_utils import universal_chat_model, make_scorer_with_llm

if __name__ == "__main__":

    EVAL_DATASET_NAME = input("Enter the evaluation dataset name (default myegpt-22Dec25):") or "myegpt-22Dec25"
    LLM_MODEL_ID = input(f"Enter LLM model ID (default {os.environ.get('MODEL_ID')}):") or os.environ.get("MODEL_ID")
    EVAL_MODEL_ID = input(f"Enter the evaluation model ID (default {os.environ.get('EVAL_MODEL_ID')}):") or os.environ.get("EVAL_MODEL_ID")
    LANGSMITH_PROJECT = input(f"Enter the Langsmith Project Name (default {os.environ.get('LANGSMITH_PROJECT')}):") or os.environ.get("LANGSMITH_PROJECT")
    os.chdir(os.path.join(os.path.dirname(__file__), "../responses/microdocs",EVAL_DATASET_NAME))
    
    # Load the responses JSON file to evaluate
    eval_file = input("Enter the path to the response JSON file:").replace('"', '').replace("'", '')
    with open(eval_file, 'r') as f:
        examples_to_evaluate = json.load(f)

    # 1. Create and/or select your dataset
    ls_client = Client()

    llm = universal_chat_model(EVAL_MODEL_ID)

    def target(inputs: dict) -> dict:
        global examples_to_evaluate
        # Find matching input and return the output
        for example in tqdm(examples_to_evaluate):
            if example.get('input').get('question') == inputs['question']:
                return {'answer': example.get('output', 'key does not exist')}
        
        # Return empty if no match found
        return {'answer':'(No response)'}

    scorer = make_scorer_with_llm(llm)
    
    # 4. Run evaluation
    experiment = ls_client.evaluate(
        target,
        data=EVAL_DATASET_NAME,
        evaluators=[scorer],
        num_repetitions=1,
        experiment_prefix=LANGSMITH_PROJECT,
        metadata={'llm_model': LLM_MODEL_ID,'eval_model': EVAL_MODEL_ID},
        upload_results=True,
    )