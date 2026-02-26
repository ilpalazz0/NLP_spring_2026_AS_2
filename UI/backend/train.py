import csv
import os
import pickle
import random
import sys
from collections import Counter

from model import (
    load_data,
    split,
    build_vocab,
    make_features,
    get_predictions,
    precision_recall_f1,
    confusion_matrix,
    mcnemar,
    NaiveBayes,
    LogisticRegression,
)

random.seed(42)

DATASET_PATH = "data/dataset.csv"
SAVE_DIR     = "saved"
TRAIN_RATIO  = 0.8
MIN_FREQ     = 2

print("Loading data…")
data = load_data(DATASET_PATH)

if len(data) == 0:
    print("ERROR: No valid rows found. Check your CSV path and column names.")
    sys.exit(1)

train_raw, test_raw = split(data, ratio=TRAIN_RATIO)
print(f"Train: {len(train_raw)}  |  Test: {len(test_raw)}")

vocab = build_vocab(train_raw, min_freq=MIN_FREQ)
print(f"Vocabulary size (stemmed, min_freq={MIN_FREQ}): {len(vocab)}")

train     = [(make_features(t, vocab, binary=False), y) for t, y in train_raw]
test      = [(make_features(t, vocab, binary=False), y) for t, y in test_raw]
train_bin = [(make_features(t, vocab, binary=True),  y) for t, y in train_raw]
test_bin  = [(make_features(t, vocab, binary=True),  y) for t, y in test_raw]

print("\nTraining models…")
nb  = NaiveBayes();         nb.train(train)
bnb = NaiveBayes();         bnb.train(train_bin)
lr  = LogisticRegression(); lr.train(train)
print("Training complete.")

gold     = [y for _, y in test]
pred_nb  = get_predictions(nb,  test)
pred_bnb = get_predictions(bnb, test_bin)
pred_lr  = get_predictions(lr,  test)

def print_metrics(name, preds, gold_labels):
    acc = sum(p == g for p, g in zip(preds, gold_labels)) / len(gold_labels)
    p, r, f1 = precision_recall_f1(preds, gold_labels)
    tp, tn, fp, fn = confusion_matrix(preds, gold_labels)
    sep = "─" * 45
    print(f"\n{sep}\n  {name}\n{sep}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  Confusion matrix  (1=pos / 0=neg):")
    print(f"            Pred+   Pred-")
    print(f"  Actual+   {tp:5d}   {fn:5d}   (TP / FN)")
    print(f"  Actual-   {fp:5d}   {tn:5d}   (FP / TN)")
    return acc

print("\n" + "═"*45)
print("  EVALUATION RESULTS  (1=positive, 0=negative)")
print("═"*45)
acc_nb  = print_metrics("Naive Bayes (count BoW)",          pred_nb,  gold)
acc_bnb = print_metrics("Binary Naive Bayes",               pred_bnb, gold)
acc_lr  = print_metrics("Logistic Regression (count BoW)",  pred_lr,  gold)

print("\n" + "═"*45)
print("  STATISTICAL SIGNIFICANCE (McNemar's Test)")
print("═"*45)
mcnemar(pred_nb,  pred_lr,  gold, label1="NB",  label2="LR")
mcnemar(pred_bnb, pred_lr,  gold, label1="BNB", label2="LR")
mcnemar(pred_nb,  pred_bnb, gold, label1="NB",  label2="BNB")

accs  = {"Naive Bayes": acc_nb, "Binary Naive Bayes": acc_bnb, "Logistic Regression": acc_lr}
best  = max(accs, key=accs.get)
print(f"\n{'═'*45}\n  SUMMARY\n{'═'*45}")
for name, acc in accs.items():
    marker = " ← best" if name == best else ""
    print(f"  {name:<28} {acc:.4f}{marker}")

os.makedirs(SAVE_DIR, exist_ok=True)

with open(f"{SAVE_DIR}/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open(f"{SAVE_DIR}/nb.pkl",    "wb") as f:
    pickle.dump(nb, f)
with open(f"{SAVE_DIR}/bnb.pkl",   "wb") as f:
    pickle.dump(bnb, f)
with open(f"{SAVE_DIR}/lr.pkl",    "wb") as f:
    pickle.dump(lr, f)

metrics = {
    "nb":  {"accuracy": acc_nb,  "precision": precision_recall_f1(pred_nb,  gold)[0],
            "recall":   precision_recall_f1(pred_nb,  gold)[1],
            "f1":       precision_recall_f1(pred_nb,  gold)[2]},
    "bnb": {"accuracy": acc_bnb, "precision": precision_recall_f1(pred_bnb, gold)[0],
            "recall":   precision_recall_f1(pred_bnb, gold)[1],
            "f1":       precision_recall_f1(pred_bnb, gold)[2]},
    "lr":  {"accuracy": acc_lr,  "precision": precision_recall_f1(pred_lr,  gold)[0],
            "recall":   precision_recall_f1(pred_lr,  gold)[1],
            "f1":       precision_recall_f1(pred_lr,  gold)[2]},
    "vocab_size": len(vocab),
    "train_size": len(train_raw),
    "test_size":  len(test_raw),
}
with open(f"{SAVE_DIR}/metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print(f"\nAll models and metrics saved to '{SAVE_DIR}/'")
print("You can now start the API with:  uvicorn api:app --reload")