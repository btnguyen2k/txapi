import sys
script_name = sys.argv[0]
args = sys.argv[1:]
if len(args) > 0:
    model_name = args[0]
    print("Loading model <" + model_name + "> from HuggingFace Hub...")
    import models
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=models.hf_cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=models.hf_cache_dir)
    print("Done.")
else:
    print(f"Usage: python {script_name} <model_name>")
    print(f"Example: python {script_name} sentence-transformers/multi-qa-mpnet-base-cos-v1")
