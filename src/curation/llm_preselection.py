import pandas as pd
from src.curation.prompts import SYSTEM_PROMPT, USER_TEMPLATE

def score_indicators_with_llm(
    indicators_df: pd.DataFrame,
    llm_client,
) -> pd.DataFrame:
    results = []

    for _, row in indicators_df.iterrows():
        prompt = USER_TEMPLATE.format(
            name=row["indicator_name"],
            description=row["description"]
        )

        response = llm_client.chat(
            system=SYSTEM_PROMPT,
            user=prompt,
            temperature=0
        )

        results.append({
            "indicator_id": row["indicator_id"],
            "name": row["indicator_name"],
            "relevance_score": response["relevance_score"],
            "justification": response["justification"]
        })

    return pd.DataFrame(results)
