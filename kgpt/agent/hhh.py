from huggingface_hub import InferenceClient

HF_TOKEN = "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"
model_id = "sentence-transformers/all-MiniLM-L6-v2"

client = InferenceClient(model=model_id, token=HF_TOKEN)

text = "Quantum mechanics deals with the smallest particles."

# Send to hosted model (it will use the right pipeline automatically)
embedding = client.feature_extraction(text)
print("✅ Remote embedding vector size:", len(embedding))
