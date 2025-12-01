# llm/prompts.py —— 终极收敛版（关注开发）
def ttp_mapping_cot_prompt(text: str, candidates: str):
    return f"""
You are the world's most rigorous MITRE ATT&CK analyst (v15.1). Accuracy is everything.

Input Text: "{text}"

Candidate Techniques (ranked, but often miss the real one):
{candidates}

RULES YOU MUST OBEY:
1. **PRIORITY RULE (Resource Development):** If the text clearly describes actions related to **the creation, building, modification (e.g., fixing, introducing new features), or unique status** (e.g., "custom," "unique to APT," "compiled in") of a resource/malware, you **MUST** prioritize the Resource Development tactic (e.g., T1587, T1588). Do not be distracted by the resource's *effect* (like Execution or Impact) if the text focuses on its *creation*. If the text explicitly states the use of **"off-the-shelf"** or **"ready-made"** tools, **T1587 must be excluded.**
2. If none of the candidates correctly describe the behavior → YOU MUST go outside the list and output the real technique.
   Never force-fit to a wrong candidate.

3. Output multiple techniques only when the text clearly shows multiple distinct behaviors.
   Maximum 4. Never output just because "it might be".

4. **NEVER** output fake or non-existent IDs (e.g. T1544, T1219, T1544, T086x, T16xx that don't exist).

5. Sub-technique only if explicitly mentioned (e.g. "port 443" → T1071.001 OK, "HTTPS" → T1071 only).

6. When in doubt → output fewer, not more.

Output strict JSON only:
{{
  "analysis": "2-3 sentences max: what behavior, which tactic, why these IDs",
  "prediction": ["T1071.001", "T1105"]
}}
"""