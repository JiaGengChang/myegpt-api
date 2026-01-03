from langchain.chat_models.base import BaseChatModel
from openevals.llm import create_llm_as_judge

from prompts import CORRECTNESS_PROMPT

def universal_chat_model(MODEL_ID: str) -> BaseChatModel:
    # Create a langchain chat model given a string MODEL_ID
    if not MODEL_ID:
        raise ValueError("MODEL_ID environment variable is not set")
    elif MODEL_ID.startswith("gpt-"):
        from langchain_openai import ChatOpenAI as ChatModel
    elif MODEL_ID.startswith("claude"):
        from langchain_anthropic import ChatAnthropic as ChatModel
    elif MODEL_ID.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI as ChatModel
    else:
        f"Defaulting to Bedrock as {MODEL_ID} does not match known providers"
        from langchain_aws import ChatBedrockConverse as ChatModel
    
    try:
        # Google, Claude, OpenAI use "model" parameter
        llm = ChatModel(
            model=MODEL_ID,
            temperature=0.,
            max_tokens=5000,
        )
    except Exception as e:
        # AWS Bedrock uses "MODEL_ID" parameter
        try: 
            llm = ChatModel(
                MODEL_ID=MODEL_ID,
                temperature=0.,
                max_tokens=5000,
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize chat model with MODEL_ID {MODEL_ID}: {e}")
    
    return llm

# used by eval/reeval.py
def make_scorer_with_llm(llm: BaseChatModel):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge=llm,
        feedback_key="score",
        continuous=True,
    )
    # a correctness score from 0 to 1, where 1 is the best
    def scorer(inputs: dict, outputs: dict, reference_outputs: dict):
        eval_result = evaluator(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs
        )
        return eval_result
    return scorer
