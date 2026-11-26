# -*- coding: utf-8 -*-
"""
download_data.py — 自动下载 alpaca-gpt4-data-zh 数据集

使用 HuggingFace datasets 库，支持 hf-mirror.com 端点。
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download

DATASET = "llm-wizard/alpaca-gpt4-data-zh"
FILENAME = "alpaca_gpt4_data_zh.json"
OUT_DIR = "datasets"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = hf_hub_download(
        repo_id=DATASET,
        filename=FILENAME,
        repo_type="dataset",
        local_dir=OUT_DIR,
    )
    print(f"Downloaded to: {path}")

if __name__ == "__main__":
    main()
