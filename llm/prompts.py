# llm/prompts.py
def behavior_extraction_prompt(text: str):
    return f"""
You are a cyber threat intelligence analyst.

Extract the atomic attacker behaviors from the text. 
NO MITRE IDs. Only behaviors.

Return JSON ONLY:

{{
  "behaviors": ["...", "..."]
}}
ONLY output valid JSON.
NO markdown code block.
NO explanation.

Text:
{text}
"""


def ttp_mapping_prompt1(text: str, behaviors, candidates):
    return f"""
You are a senior MITRE ATT&CK analyst.

For EACH behavior, evaluate ALL candidate techniques.
For each candidate, provide a relevance score from 0 to 1.

Rules:
- You MUST only select techniques from the candidate list.
- Score meaning:
  1.0 = exact match
  0.7 = partial match
  0.3 = weak match
  0.0 = unrelated
- After scoring, choose the technique with the highest score.
-

Output JSON ONLY:

{{
  "thinking": [
    "Behavior X → score table … → best choice",
    ...
  ],
  "techniques": ["Txxxx", "Txxxx"]
  
}}

Behaviors:
{behaviors}

Candidates:
{candidates}

Text:
{text}
ONLY output valid JSON.
NO markdown code block.
NO explanation.
"""
def ttp_mapping_prompt(text: str, behaviors, candidates):
    return f"""
You are a senior MITRE ATT&CK analyst.

For EACH behavior, evaluate ALL candidate techniques and provide the final prediction.
If multiple behaviors together indicate a single combined TTP, you must also consider that.

Rules:
- You MAY reference candidates, but you are NOT restricted to them..
- Score meaning:
  1.0 = exact match
  0.7 = partial match
  0.3 = weak match
  0.0 = unrelated
- After scoring all candidate techniques for a behavior, base the techniques with the score to find final predictions.
- Provide the reasoning for the selected predictions.

Output JSON ONLY:

{{
  "thinking": [
    "For behavior X , based on the analysis of all candidates, the best matching technique is Txxxx with score Y. Reasoning: ...",
    ...
  ],
   "techniques": ["Txxxx", "Txxxx"]
  "prediction": ["Txxxx"]
}}

Behaviors:
{behaviors}

Candidates:
{candidates}

Text:
{text}
ONLY output valid JSON.
NO markdown code block.
NO explanation.
"""
