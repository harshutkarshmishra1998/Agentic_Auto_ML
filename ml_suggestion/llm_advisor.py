import json
import api_keys
from groq import Groq

# ensures .env is loaded and GROQ_API_KEY exists
import api_keys  # noqa: F401

LLM_MODEL = "llama-3.1-8b-instant"


# -----------------------------------------------------
# CLIENT
# -----------------------------------------------------

def _get_client():
    return Groq()


# -----------------------------------------------------
# PROMPT BUILDER
# -----------------------------------------------------

def _build_prompt(fingerprint, task_type, scores):
    return f"""
You are an expert AutoML system.

Your task:
Adjust model suitability scores based on dataset structure.

DATASET FINGERPRINT:
{fingerprint}

TASK TYPE:
{task_type}

CURRENT MODEL SCORES:
{scores}

Instructions:
Return ONLY valid JSON.

Each model gets a multiplier between 0.8 and 1.2

Rules example:
- strong nonlinearity → boost tree models
- very small dataset → penalize neural networks
- high categorical ratio → boost tree models
- high correlation → penalize linear models
- severe imbalance → boost boosting models

Output format:
{{
  "model_name": multiplier
}}
"""


# -----------------------------------------------------
# PARSE RESPONSE
# -----------------------------------------------------

def _safe_parse_json(text: str):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {}


# -----------------------------------------------------
# APPLY ADJUSTMENTS
# -----------------------------------------------------

def _apply_adjustments(scores, multipliers):
    adjusted = {}

    for model, score in scores.items():
        m = multipliers.get(model, 1.0)
        adjusted[model] = score * float(m)

    return adjusted


# -----------------------------------------------------
# PUBLIC ENTRY
# -----------------------------------------------------

def maybe_refine_with_llm(fingerprint, task_type, scores, use_llm=False):
    if not use_llm:
        return scores

    try:
        client = _get_client()

        prompt = _build_prompt(fingerprint, task_type, scores)

        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        response_text = completion.choices[0].message.content

        # print("CALLING LLM...")
        # print("PROMPT SENT")
        # print(prompt)

        # print("RAW LLM RESPONSE:")
        # print(response_text)
        
        multipliers = _safe_parse_json(response_text) #type: ignore

        if not multipliers:
            return scores

        llm_scores = _apply_adjustments(scores, multipliers)

        # fusion
        final_scores = {}
        for m in scores:
            # final_scores[m] = 0.75 * scores[m] + 0.25 * llm_scores[m]
            # ---------------------------------------
            # adaptive fusion weight
            # ---------------------------------------
            llm_weight = 0.25

            if fingerprint.complexity_score == 0:
                llm_weight = 0.4

            if fingerprint.feature_correlation == 0:
                llm_weight = max(llm_weight, 0.35)

            det_weight = 1 - llm_weight

            final_scores = {}
            for m in scores:
                final_scores[m] = det_weight * scores[m] + llm_weight * llm_scores[m]

        return final_scores

    except Exception as e:
        import traceback
        print("\n🚨 LLM CALL FAILED")
        print("ERROR:", e)
        traceback.print_exc()
        return scores