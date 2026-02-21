from groq import Groq
import json
import api_keys

MODEL = "llama-3.3-70b-versatile"


def _client():
    return Groq()


def build_prompt(summary):
    return f"""
You are an expert ML system designer.

Select appropriate model families for this dataset.

DATASET SUMMARY:
{json.dumps(summary, indent=2)}

Return STRICT JSON:

problem_type
problem_confidence (0-1)
recommended_models (ordered best first)
reasoning
model_dependent_preprocessing
"""


def validate_llm_output(result):

    required = ["problem_type", "recommended_models"]

    for r in required:
        if r not in result:
            raise ValueError(f"LLM missing field: {r}")

    result.setdefault("problem_confidence", 0.5)
    result.setdefault("model_dependent_preprocessing", [])

    return result


def llm_model_selection(dataset_summary):

    client = _client()

    completion = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[{"role": "user", "content": build_prompt(dataset_summary)}],
        response_format={"type": "json_object"}
    )

    result = json.loads(completion.choices[0].message.content) #type: ignore
    return validate_llm_output(result)