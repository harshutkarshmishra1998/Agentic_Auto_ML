import json
import os
from typing import Tuple
import api_keys
from groq import Groq


# Allowed roles in system
VALID_ROLES = {
    "numeric_continuous",
    "numeric_discrete",
    "categorical_nominal",
    "categorical_ordinal",
    "identifier",
    "datetime",
    "text_freeform",
    "unknown",
}


# -----------------------------
# Groq client
# -----------------------------
def _get_client():
    return Groq()


# -----------------------------
# Prompt builder
# -----------------------------
def _build_prompt(column_name, profile, current_role):

    return f"""
You are a schema inference expert.

Your job is to determine the TRUE semantic role of a dataset column.

COLUMN NAME:
{column_name}

CURRENT ROLE (rule-based guess):
{current_role}

COLUMN STATISTICS:
dtype: {profile.dtype}
n_unique: {profile.n_unique}
unique_ratio: {profile.unique_ratio}
missing_ratio: {profile.missing_ratio}
is_numeric: {profile.is_numeric}
is_integer_like: {profile.is_integer_like}
mean: {profile.mean}
std: {profile.std}
min: {profile.min_val}
max: {profile.max_val}

SAMPLE VALUES:
{profile.sample_values}

Choose ONE role from this list:
numeric_continuous
numeric_discrete
categorical_nominal
categorical_ordinal
identifier
datetime
text_freeform
unknown

IMPORTANT:
Respond ONLY JSON.

FORMAT:
{{
  "role": "...",
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}
"""


# -----------------------------
# Response parser
# -----------------------------
def _parse_response(text: str):

    text = text.strip()

    # remove markdown if model returns ```json
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    data = json.loads(text)

    role = data.get("role")
    confidence = float(data.get("confidence", 0.5))

    if role not in VALID_ROLES:
        raise ValueError("Invalid role from LLM")

    confidence = max(0.0, min(1.0, confidence))

    return role, confidence


# -----------------------------
# PUBLIC FUNCTION
# -----------------------------
def resolve_with_llm(column_name, profile, current_role) -> Tuple[str, float]:
    """
    Uses Groq LLM to resolve ambiguous column role.

    Returns:
        role, confidence
    """

    try:
        client = _get_client()

        prompt = _build_prompt(column_name, profile, current_role)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # fast + strong reasoning
            temperature=0,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content

        role, confidence = _parse_response(content) #type: ignore

        return role, confidence

    except Exception as e:
        # NEVER break pipeline — fallback
        print(f"[LLM RESOLVER WARNING] {column_name}: {e}")
        return current_role, 0.5