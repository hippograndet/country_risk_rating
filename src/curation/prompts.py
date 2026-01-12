SYSTEM_PROMPT = """
You are an expert macroeconomic risk analyst.
Your task is to assess whether economic indicators are relevant
for predicting sovereign risk ratings.
"""

USER_TEMPLATE = """
Target:
OECD country risk rating (0–7), reflecting economic, financial,
and political risk.

Indicator:
Name: {name}
Description: {description}

Question:
Is this indicator relevant for predicting country risk?
Respond with:
- relevance_score: integer from 0 (irrelevant) to 5 (highly relevant)
- short justification (1 sentence)
"""
