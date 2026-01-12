import pandas as pd

def select_top_candidates(
    scored_df: pd.DataFrame,
    min_score: int = 3
) -> pd.DataFrame:
    return (
        scored_df
        .query("relevance_score >= @min_score")
        .sort_values("relevance_score", ascending=False)
    )
