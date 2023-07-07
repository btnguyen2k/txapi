# Datasets

> The data in these datasets has been collected from public sources. The data is provided "as is", without warranty or any representation of accuracy, timeliness, or completeness.

Dataset file format: `dataset_<source>_<number_of_records>_<data_type>.jsonl`

- `<source>`: source of the data, e.g. `vietnamnet`, `vnexpress`.
- `<data_type>`:
  - `news_title_summary`: each record contains the title and summary of a news article collected from `<source>`.

Available datasets (as of 2023-07-08):
- `dataset_vietnamnet_323_news_title_summary.jsonl`: 323 news articles collected from [vietnamnet.vn](https://vietnamnet.vn) (language: Vietnamese).
- `dataset_vnexpress_691_news_title_summary.jsonl`: 691 news articles collected from [vnexpress.net](https://vnexpress.net) (language: Vietnamese).

> **Use these datasets for research and educational purposes only!**

# Embedding benchmark results (as of 2023-07-08)

Benchmark result for dataset `dataset_vietnamnet_323_news_title_summary.jsonl`:

| model (sentence-transformers) | speed (*) |       min |       max |      mean |       p80 |       p90 |       p95 |       p99 |
|:------------------------------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| all-mpnet-base-v2             |      7.21 |     0.274 | **0.967** |     0.772 |     0.695 |     0.631 |     0.588 |     0.448 |
| all-MiniLM-L6-v2              |     32.61 |     0.272 |     0.921 |     0.720 |     0.658 |     0.612 |     0.562 |     0.415 |
| all-MiniLM-L12-v2             |     17.00 |     0.273 |     0.950 |     0.705 |     0.619 |     0.563 |     0.518 |     0.424 |
| multi-qa-mpnet-base-cos-v1    |      7.04 |     0.341 |     0.950 |     0.778 |     0.711 |     0.649 |     0.602 |     0.469 |
| multi-qa-MiniLM-L6-cos-v1     |     32.78 |     0.244 |     0.946 |     0.722 |     0.647 |     0.597 |     0.571 |     0.508 |
| multi-qa-distilbert-cos-v1    |     13.50 | **0.428** |     0.934 | **0.783** | **0.721** | **0.671** | **0.627** | **0.520** |

> Overall `multi-qa-distilbert-cos-v1` has the highest accuracy on this dataset while maintaining a balanced performance.

(*) Speed is the number of records processed per second, subject to the hardware used for benchmarking.

Benchmark result for dataset `dataset_vnexpress_691_news_title_summary.jsonl`:

| model (sentence-transformers) | speed (*) |       min |       max |      mean |       p80 |       p90 |       p95 |       p99 |
|:------------------------------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| all-mpnet-base-v2             |      7.53 |     0.172 | **0.955** |     0.724 |     0.645 | **0.586** |     0.529 |     0.375 |
| all-MiniLM-L6-v2              |     34.02 |     0.227 |     0.915 |     0.668 |     0.585 |     0.529 |     0.469 |     0.363 |
| all-MiniLM-L12-v2             |     17.72 |     0.162 |     0.903 |     0.647 |     0.554 |     0.497 |     0.435 |     0.336 |
| multi-qa-mpnet-base-cos-v1    |      7.55 |     0.229 |     0.945 |     0.720 |     0.630 |     0.570 |     0.508 |     0.374 |
| multi-qa-MiniLM-L6-cos-v1     |     34.15 |     0.160 |     0.898 |     0.670 |     0.582 |     0.524 |     0.467 |     0.367 |
| multi-qa-distilbert-cos-v1    |     14.67 | **0.265** |     0.939 | **0.731** | **0.655** |   _0.585_ | **0.542** | **0.416** |

> Overall `multi-qa-distilbert-cos-v1` has the highest accuracy on this dataset while maintaining a balanced performance.

(*) Speed is the number of records processed per second, subject to the hardware used for benchmarking.

min/max/mean/p80/p90/p95/p99 are the cosine similarity values between the title and the summary of a news article.
