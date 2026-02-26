import json
from ngram_engine import NgramModel

DATA_DIR = "data"
MODES    = ["none", "laplace", "k", "interpolation", "backoff", "kneser_ney"]

print("Loading corpus and building model...")
print("(EM for interpolation + kneser_ney precompute may take ~1-2 min)")
model = NgramModel(DATA_DIR)

stats = model.stats()
print(f"\nCorpus stats: {json.dumps(stats)}")
print(f"Test corpus (hatai-1.md): {len(model.test):,} tokens\n")

print("Computing perplexities on hatai-1.md ...")
results = {}
for mode in MODES:
    print(f"  {mode} ...", end=" ", flush=True)
    uni = model.perplexity_unigram(mode)
    bi  = model.perplexity_bigram(mode)
    tri = model.perplexity_trigram(mode)
    results[mode] = {
        "unigram":  round(uni, 4) if uni < 1e8 else None,
        "bigram":   round(bi,  4) if bi  < 1e8 else None,
        "trigram":  round(tri, 4) if tri < 1e8 else None,
    }
    print("done")

print("\n" + "="*60)
print("PASTE THE BLOCK BELOW INTO api.py")
print("="*60 + "\n")
print("# ── Pre-computed n-gram results (run precompute_ngrams.py to regenerate) ──")
print(f"NGRAM_STATS = {json.dumps(stats, indent=4)}\n")
print(f"NGRAM_PERPLEXITY = {json.dumps(results, indent=4)}")
print("\n" + "="*60)