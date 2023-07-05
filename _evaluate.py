cache_dir = "./cache_hf"
#models = ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2"]
#models = ["sentence-transformers/multi-qa-mpnet-base-cos-v1", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "sentence-transformers/multi-qa-distilbert-cos-v1"]
models = [
    "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/multi-qa-mpnet-base-cos-v1", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "sentence-transformers/multi-qa-distilbert-cos-v1",
]
for model_name in models:
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

    very_long_text = "This is a very long text. " * 2048
    encoded_input = tokenizer(very_long_text, padding=True, truncation=True, return_tensors='pt')
    print(f"Model       : {model_name}")
    print(f"Token length: {len(encoded_input['input_ids'][0])}")

    # query = "Does DO CMS support Markdown?"
    # docs = [
    #     "DO CMS is a Content Management System that helps authors publish website content through a CI/CD flow. Unlike other CMS, there is no UI to create, update and publish content in DO CMS. Instead, website content is built and published via CI/CD pipelines.",
    #     "Authoring website content with DO CMS is as similar to daily work of an developer: push (documents) to Git, make pull requests, trigger CI/CD pipelines, build and containerize website content and finally deploy it for end users to access.",
    #     "DO CMS supports GitHub Flavored Markdown (GFM), which is the dialect of Markdown that is currently supported for user content on GitHub.com and GitHub Enterprise. Furthermore, Mathematical and Chemical formulas are supported."
    # ]
    query = "DO CMS có hỗ trợ Markdown không?"
    docs = [
        "DO CMS là Hệ thống quản lý nội dung giúp tác giả xuất bản nội dung trang web thông qua luồng CI/CD. Sẽ không có giao diện để người dùng tạo, cập nhật và xuất bản nội dung lên trang web. Thay vào đó, nội dung của trang web sẽ được xây dựng và xuất bản thông qua qui trình CI/CD.",
        "Biên tập nội dung trang web với DO CMS sẽ rất giống với công việc hàng ngày của một lập trình viên: đẩy (tài liệu) lên Git, tạo pull request, kích hoạt CI/CD, 'biên dịch' và đóng gói nội dung trang web và cuối cùng là triển khai đến người dùng cuối.",
        "DO CMS hỗ trợ GitHub Flavored Markdown (GFM), là một phương ngữ của Markdown hiện được hỗ trợ trên GitHub.com và phiên bản GitHub Enterprise. Ngoài ra, các công thức Toán học và Hoá học cũng được hỗ trợ.",
    ]
    import models
    query_emb = models.encode_embeddings(model, tokenizer, [query])
    docs_emb = models.encode_embeddings(model, tokenizer, docs)
    scores = torch.mm(query_emb, docs_emb.transpose(0, 1))[0].cpu().tolist()
    doc_score_pairs = list(zip(docs, scores))

    print(f"Query       : {query}")
    for doc, score in doc_score_pairs:
        print(f"{score:.10f}: {doc[:36]}...")
    print("="*70)
