import uvicorn

if __name__ == "__main__":
    import os
    num_workers = int(os.getenv("NUM_WORKERS", 1))
    if num_workers <= 1:
        num_workers = 1
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=num_workers)
