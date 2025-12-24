from langsmith import Client

import os
from dotenv import load_dotenv
assert load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from langchain_openai import ChatOpenAI
import json

if __name__ == "__main__":

    os.chdir(os.path.join(os.path.dirname(__file__), "../responses/microdocs"))

    EVAL_DATASET_NAME = input("Enter the evaluation dataset name: ") or "myegpt-22Dec25"
    LLM_MODEL_ID = input(f"Enter LLM model ID: (default) {os.environ.get('MODEL_ID')}") or os.environ.get("MODEL_ID")
    EVAL_MODEL_ID = input(f"Enter the evaluation model ID: (default) {os.environ.get('EVAL_MODEL_ID')}") or os.environ.get("EVAL_MODEL_ID")
    OUTPUT_JSON = input(f"Enter the output JSON file name: (default) {EVAL_MODEL_ID}-{LLM_MODEL_ID}.json") or f"EMBED_{EVAL_MODEL_ID}_LLM_{LLM_MODEL_ID}.json"

    # 1. Create and/or select your dataset
    ls_client = Client()

    llm = ChatOpenAI(
        model=EVAL_MODEL_ID,
        temperature=0.,
    )

    def target(inputs: dict) -> dict:
        global examples_to_evaluate
        print('dataset input: ',inputs)
        # Find matching input and return the output
        for example in examples_to_evaluate:
            if example.get('inputs').get('question') == inputs['question']:
                print(example)
                return {'answer': example.get('output', 'key does not exist')}
        
        # Return empty if no match found
        return {'answer':'I couldn\'t find the answer'}

    # a correctness score from 0 to 1, where 1 is the best
    def scorer(inputs: dict, outputs: dict, reference_outputs: dict):
        evaluator = create_llm_as_judge(
            prompt=CORRECTNESS_PROMPT,
            judge=llm,
            feedback_key="score",
            continuous=True,
        )
        eval_result = evaluator(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs
        )
        return eval_result

    # Load the failed examples JSON file
    eval_file = input("Enter the path to the response JSON file:")
    with open(eval_file, 'r') as f:
        examples_to_evaluate = json.load(f)
    
    # 4. Run evaluation
    experiment = ls_client.evaluate(
        target,
        data=EVAL_DATASET_NAME,
        evaluators=[scorer],
        experiment_prefix=os.environ.get("LANGSMITH_PROJECT"),
        metadata={
                    'llm_model': LLM_MODEL_ID,
                    'eval_model': EVAL_MODEL_ID,
                },
        upload_results=True
    )

    # 5. Export results to a JSON file
    df = experiment.to_pandas()
    df[['inputs.question',
        'reference.answer',
        'reference.scoring',
        'outputs.answer',
        'feedback.score']]\
            .to_json(OUTPUT_JSON,orient='index',indent=2)