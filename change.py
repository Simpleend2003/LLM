# clean_mitre_dataset_plus.py
# 一键清洗 + 每种 TTP 只保留一行

import pandas as pd
import re
import random

def clean_question(text):
    """去掉 Please help to identify... 前缀"""
    if pd.isna(text):
        return ""
    prefixes = [
        r"Please help to identify the following description belonging to which technique in MITRE and the corresponding tactics[:\s]*",
        r"Please help to identify the following description belonging to which technique in MITRE and the corresponding tactics\s*[:\.]*",
    ]
    cleaned = str(text)
    for p in prefixes:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def extract_techniques(answer):
    """提取所有 Txxxx 或 Txxxx.xxx"""
    if pd.isna(answer):
        return []
    matches = re.findall(r'T\d{4}(?:\.\d{3})?', str(answer))
    # 去重但保持原始顺序
    seen = set()
    unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique

# ==================== 主程序 ====================
input_csv = "cyber_MITRE_CTI_dataset .csv"        # 改成你的原始文件名
full_output = "data/cleaned_full.csv"
one_per_ttp_output = "data/one_per_ttp.csv"

print("正在读取数据...")
df = pd.read_csv(input_csv)

print("正在清洗 question...")
df['question'] = df['question'].apply(clean_question)

print("正在提取 TTP...")
df['answer'] = df['answer'].apply(extract_techniques)

# 保存完整清洗版
df[['question', 'answer']].to_csv(full_output, index=False)
print(f"完整清洗版已保存：{full_output}，共 {len(df)} 条")

# ==================== 每种 TTP 只保留一行 ====================
print("正在生成 every TTP only one sample...")
ttp_to_rows = {}

for idx, row in df.iterrows():
    question = row['question']
    techniques = row['answer']
    if not techniques:
        continue
    # 把多个 TTP 的都算进去
    for ttp in techniques:
        if ttp not in ttp_to_rows:
            ttp_to_rows[ttp] = []
        ttp_to_rows[ttp].append({
            'question': question,
            'answer': [ttp]  # 只保留当前这个 TTP
        })

# 每种 TTP 随机挑一条（或第一条）
selected_rows = []
for ttp, rows in ttp_to_rows.items():
    chosen = random.choice(rows)  # 随机挑，公平
    # 也可以用 rows[0] 取第一条
    selected_rows.append({
        'question': chosen['question'],
        'answer': str(chosen['answer'])  # 存成字符串 ['Txxxx']
    })

one_per_df = pd.DataFrame(selected_rows)
one_per_df = one_per_df[['question', 'answer']]
one_per_df.to_csv(one_per_ttp_output, index=False)

print(f"每种 TTP 只保留一行 已保存：{one_per_ttp_output}")
print(f"共 {len(one_per_df)} 种唯一 TTP")

print("\n全部完成！")
print("   - 完整数据集 → cleaned_full.csv")
print("   - 每种 TTP 一条 → one_per_ttp.csv")