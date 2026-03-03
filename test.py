from src.retrieval.embedder import ContractEmbedder
e = ContractEmbedder()
results = e.search("What does Chapter 1 say?", k=3)
for r in results:
    print(f"Score: {r['score']:.4f} | {r['text'][:80]}")