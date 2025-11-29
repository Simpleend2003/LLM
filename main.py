# main.py

import pandas as pd
import json
import time
from mitre.knowledge_base import MITREKnowledgeBase
from mitre.rag_retriever import RAGRetriever
from llm.ttp_extractor import TTPExtractor
from evaluator.metrics import calculate_coverage
import os
# 设置代理（保留不变）
os.environ['HTTP_PROXY'] = "127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "127.0.0.1:7890"

def parse_ttp_list(s):
    if pd.isna(s) or not s:
        return []
    s = s.strip("[]").replace("'", "").replace('"', "")
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    print(1)
    kb = MITREKnowledgeBase()
    print(2)
    retriever = RAGRetriever(kb)
    print(3)
    extractor = TTPExtractor(retriever)

    df = pd.read_csv("data/test_tram.csv")
    results = []

    for idx, row in df.iterrows():
        print(f"Processing {idx+1}/{len(df)}")

        text = row["text1"]
        labels = parse_ttp_list(row["labels"])

        check, thinking, related = extractor.extract(text)
        coverage = calculate_coverage(labels, check)

        results.append({
            "text1": text,
            "labels": labels,
            "check": check,
            "think": thinking,
            "related_techniques": json.dumps(related, ensure_ascii=False),
            "score": (
                f"Full{coverage['full_coverage']}, "
                f"Semi{coverage['semi_coverage']}, "
                f"No{coverage['no_coverage']}, "
                f"FP{coverage['false_positive']}"
            ),
            **coverage
        })

        time.sleep(0.5)

    out_df = pd.DataFrame(results)
    out_df.to_csv("output_tram.csv", index=False, encoding="utf-8-sig")
    print("Done! Saved output.csv")


if __name__ == "__main__":
    main()
