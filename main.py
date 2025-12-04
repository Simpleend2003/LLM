# main.py
import base64

import pandas as pd
import json
import time
from mitre.knowledge_base import MITREKnowledgeBase
from mitre.rag_retriever import RAGRetriever
from llm.ttp_extractor import TTPExtractor
from evaluator.metrics import calculate_coverage
import os
import csv
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

    df = pd.read_csv("data/wrongtext.csv")
    results = []
    full = 0
    semi = 0
    all = 0
    no = 0
    wrong = 0
    for idx, row in df.iterrows():
        print(f"Processing {idx + 1}/{len(df)}")

        text = row["text1"]
        labels = parse_ttp_list(row["labels"])

        check, thinking, related = extractor.extract(text)
        coverage = calculate_coverage(labels, check)
        full += coverage['full_coverage']
        semi += coverage['semi_coverage']
        no += coverage['no_coverage']
        wrong += coverage['false_positive']
        all += (coverage['full_coverage']+coverage['semi_coverage']+coverage['false_positive'])
        # --- 关键修改：对长文本字段进行 Base64 编码 ---
        results.append({
            "text1": text,
            "labels": labels,
            "check": check,
            "think": thinking[:3000],
            "related_techniques": (json.dumps(related, ensure_ascii=False))[:3000],
            "score": (
                f"Full{coverage['full_coverage']}, "
                f"Semi{coverage['semi_coverage']}, "
                f"No{coverage['no_coverage']}, "
                f"FP{coverage['false_positive']}"
            ),
            **coverage
        })
        time.sleep(0.5)
    print("full:",full)
    print("semi:",semi)
    print("no::",no)
    print("wrong:",wrong)
    print((full+semi)/all)
    out_df = pd.DataFrame(results)
    out_df.to_csv(
        "output_1.csv",
        index=False,
        encoding="utf-8-sig",
        # 强制对所有字段（包括数字和字符串）使用引号包裹
        quoting=csv.QUOTE_ALL,
        # 确保字段内部的引号正确转义（例如："think" 内部的 " 变成 ""）
        doublequote=True
    )
    print("Done! Saved output.csv")


if __name__ == "__main__":
    main()
