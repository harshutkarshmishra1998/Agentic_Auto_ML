from ml_suggestion.pipeline import suggest_models

results = suggest_models(last_n=6, use_llm=True)

for r in [results]:
    print(r)