"""
Utility script to perform embedding benchmarking.

Author: Thanh Nguyen <btnguyen2k(at)gmail(dot)com>

This script read the configuration file located at ./datasets/benchmark.conf
"""
import pandas as pd
import models
import torch
import argparse

aparser = argparse.ArgumentParser()
aparser.add_argument('--no-progress', type=bool, help='Set to disable progress bar', default=False)
args = aparser.parse_args()
display_progress = not args.no_progress


# Print iterations progress
def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    ref: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def calc_score(model, tokenizer, query: str, docs: [str]):
    query_emb = models.encode_embeddings(model, tokenizer, [query])
    docs_emb = models.encode_embeddings(model, tokenizer, docs)
    # scores = torch.mm(query_emb, docs_emb.transpose(0, 1))[0].cpu().tolist()
    scores = torch.nn.CosineSimilarity(dim=1)(query_emb, docs_emb).cpu().tolist()
    doc_score_pairs = list(zip(docs, scores))
    return doc_score_pairs


def benchmark_model(model_name: str, dataset: [str], cache_dir="./cache_hf"):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, model_max_length=1024)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

    query_arr, summary_arr, distance_arr = [], [], []
    import time
    start = time.time()
    import json
    for line in dataset:
        item = json.loads(line)
        query = item["title"].strip()
        docs = [item["summary"].strip()]
        doc_score_pairs = calc_score(model, tokenizer, query, docs)
        query_arr.append(query)
        summary_arr.append(docs[0])
        distance_arr.append(1.0 - doc_score_pairs[0][1])
        if display_progress:
            progress_bar(len(query_arr), len(dataset), prefix='Progress:', suffix='Complete ('+model_name+')', length=50)
    duration = time.time() - start
    speed = len(query_arr) / duration
    print(f"Model: {model_name} - Time elapsed: {duration:.2f} seconds, speed: {speed:.2f} items/s")
    import pandas as pd
    df = pd.DataFrame({"query": query_arr, "summary": summary_arr, "distance": distance_arr})
    min, max, mean = 1.0 - df["distance"].max(), 1.0 - df["distance"].min(), 1.0 - df["distance"].mean()
    p80, p90 = 1.0 - df["distance"].quantile(0.80), 1.0 - df["distance"].quantile(0.90)
    p95, p99 = 1.0 - df["distance"].quantile(0.95), 1.0 - df["distance"].quantile(0.99)
    print(
        f"Min: {min:.3f} / Max: {max:.3f} / Mean: {mean:.3f} / P80: {p80:.3f} / P90: {p90:.3f} / P95: {p95:.3f} / P99: {p99:.3f}")
    print("-" * 120)
    return df, speed


# load benchmark configurations from file
model_list = []
dataset_files = []
with open("./datasets/benchmark.conf", 'r') as file:
    line = file.readline()
    while line:
        line = line.strip()
        if len(line) > 0 and not line.startswith("#") and not line.startswith("//"):
            section, value = line.split("=")
            if section.strip().upper() == "MODEL":
                model_list.append(value.strip())
            if section.strip().upper() == "DATAFILE":
                dataset_files.append("./datasets/" + value.strip())
        line = file.readline()

for dataset_file in dataset_files:
    with open(dataset_file, 'r') as file:
        dataset = file.readlines()
    print(f"Data file <{dataset_file}> has {len(dataset)} lines")

    min_arr, max_arr, mean_arr = [], [], []
    p80_arr, p90_arr, p95_arr, p99_arr = [], [], [], []
    speed_arr = []
    for model_name in model_list:
        df, speed = benchmark_model(model_name, dataset)
        min_arr.append(round(1.0 - df["distance"].max(), 3))
        max_arr.append(round(1.0 - df["distance"].min(), 3))
        mean_arr.append(round(1.0 - df["distance"].mean(), 3))
        p80_arr.append(round(1.0 - df["distance"].quantile(0.80), 3))
        p90_arr.append(round(1.0 - df["distance"].quantile(0.90), 3))
        p95_arr.append(round(1.0 - df["distance"].quantile(0.95), 3))
        p99_arr.append(round(1.0 - df["distance"].quantile(0.99), 3))
        speed_arr.append(round(speed, 2))
    df = pd.DataFrame({
        "speed": speed_arr,
        "min": min_arr, "max": max_arr, "mean": mean_arr,
        "p80": p80_arr, "p90": p90_arr, "p95": p95_arr, "p99": p99_arr,
    }, index=model_list)
    print(f"Data file <{dataset_file}> benchmark result:")
    print(df.to_markdown())
    print("=" * 120)
