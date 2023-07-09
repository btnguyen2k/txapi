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

# Embedding benchmark results (as of 2023-07-09)

Benchmark result for dataset `dataset_vietnamnet_323_news_title_summary.jsonl`:

| models                                           | speed (*) |       min |       max |      mean |       p80 |       p90 |       p95 |       p99 |
|:-------------------------------------------------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| sentence-transformers/all-mpnet-base-v2          |     10.39 |     0.274 |   _0.967_ |     0.772 |     0.695 |     0.631 |     0.588 |     0.448 |
| sentence-transformers/all-MiniLM-L6-v2           |     58.05 |     0.272 |     0.921 |     0.720 |     0.658 |     0.612 |     0.562 |     0.415 |
| sentence-transformers/all-MiniLM-L12-v2          |     30.23 |     0.273 |     0.950 |     0.705 |     0.619 |     0.563 |     0.518 |     0.424 |
| sentence-transformers/multi-qa-mpnet-base-cos-v1 |     10.98 |     0.341 |     0.950 |     0.778 |     0.711 |     0.649 |     0.602 |     0.469 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1  |     58.60 |     0.244 |     0.946 |     0.722 |     0.647 |     0.597 |     0.571 |     0.508 |
| sentence-transformers/multi-qa-distilbert-cos-v1 |     21.01 |   _0.428_ |     0.934 |   _0.783_ |   _0.721_ |   _0.671_ |   _0.627_ |   _0.520_ |
| vinai/bartpho-syllable                           |      3.30 | **0.935** | **0.991** | **0.977** | **0.971** | **0.965** | **0.960** | **0.950** |

> `vinai/bartpho-syllable` has the highest accuracy on this dataset (language=Vietnamese) but it is very slow.
> 
> `multi-qa-distilbert-cos-v1` has good accuracy while maintaining a balanced performance.

(*) Speed is the number of records processed per second, subject to the hardware used for benchmarking.

min/max/mean/p80/p90/p95/p99 are the cosine similarity values between the title and the summary of a news article.


Benchmark result for dataset `dataset_vnexpress_691_news_title_summary.jsonl`:

| models                                           | speed (*) |       min |       max |      mean |       p80 |       p90 |       p95 |       p99 |
|:-------------------------------------------------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| sentence-transformers/all-mpnet-base-v2          |     12.56 |     0.172 |   _0.955_ |     0.724 |     0.645 |     0.586 |     0.529 |     0.375 |
| sentence-transformers/all-MiniLM-L6-v2           |     65.18 |     0.227 |     0.915 |     0.668 |     0.585 |     0.529 |     0.469 |     0.363 |
| sentence-transformers/all-MiniLM-L12-v2          |     33.36 |     0.162 |     0.903 |     0.647 |     0.554 |     0.497 |     0.435 |     0.336 |
| sentence-transformers/multi-qa-mpnet-base-cos-v1 |     12.56 |     0.229 |     0.945 |     0.720 |     0.630 |     0.570 |     0.508 |     0.374 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1  |     64.92 |     0.160 |     0.898 |     0.670 |     0.582 |     0.524 |     0.467 |     0.367 |
| sentence-transformers/multi-qa-distilbert-cos-v1 |     24.15 |   _0.265_ |     0.939 |   _0.731_ |   _0.655_ |   _0.585_ |   _0.542_ |   _0.416_ |
| vinai/bartpho-syllable                           |      3.71 | **0.875** | **0.992** | **0.971** | **0.963** | **0.958** | **0.951** | **0.936** |

> `vinai/bartpho-syllable` has the highest accuracy on this dataset (language=Vietnamese) but it is very slow.
> 
> `multi-qa-distilbert-cos-v1` has good accuracy while maintaining a balanced performance.

(*) Speed is the number of records processed per second, subject to the hardware used for benchmarking.

min/max/mean/p80/p90/p95/p99 are the cosine similarity values between the title and the summary of a news article.
