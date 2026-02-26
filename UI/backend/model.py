import csv
import re
import math
from collections import Counter, defaultdict
import random

try:
    from nltk.corpus import stopwords as nltk_sw
    AZ_STOPWORDS = set(nltk_sw.words('azerbaijani'))
except Exception:
    AZ_STOPWORDS = {
        "bir", "bu", "da", "də", "və", "ki", "o", "ilə", "üçün", "artıq",
        "çox", "daha", "belə", "lakin", "amma", "bütün", "heç", "ancaq",
        "isə", "ona", "onun", "onlar", "biz", "siz", "mən", "sən", "hər",
        "nə", "necə", "harada", "hansı", "nəyin", "nədən", "özü", "özünü",
    }

AZ_LOWER_MAP = str.maketrans("ƏIÖÜĞŞÇİ", "əıöüğşçi")

def tokenize(text):
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(AZ_LOWER_MAP).lower()
    text = re.sub(r"[^a-zəıöüğşç\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in AZ_STOPWORDS and len(t) > 1]
    return tokens

def load_data(path):
    data = []
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            raw_label = (row.get("sentiment") or "").strip()
            raw_text  = (row.get("review")    or "").strip()

            if not raw_label:
                print(f"  [Warning] Row {i}: missing 'sentiment' value – skipped.")
                continue
            if not raw_text:
                print(f"  [Warning] Row {i}: empty 'review' text – skipped.")
                continue

            # Accept both numeric ("0"/"1") and string ("positive"/"negative") labels
            if raw_label in ("1", "positive"):
                label = 1
            elif raw_label in ("0", "negative"):
                label = 0
            else:
                print(f"  [Warning] Row {i}: unexpected label '{raw_label}' – skipped.")
                continue

            tokens = tokenize(raw_text)
            data.append((tokens, label))

    print(f"  Loaded {len(data)} valid rows.")
    return data


POS_WORDS = {
    "yaxşı", "əla", "gözəl", "mükəmməl", "möhtəşəm", "əzəmətli",
    "maraqlı", "əyləncəli", "zövqlü", "heyrətləndirici", "qəşəng",
    "sevdim", "bəyəndim", "xoşladım", "tövsiyə", "məmnun",
    "istedadlı", "parlaq", "uğurlu", "təsirli", "fövqəladə",
    "əyləncəli", "şən", "möcüzəli", "fantastik", "sensasion",
}
NEG_WORDS = {
    "pis", "dəhşətli", "zəif", "darıxdırıcı", "acınacaqlı",
    "iyrənc", "dözülməz", "məyus", "ən pis", "boş",
    "axmaq", "gülünc", "yazıq", "sinir", "cansıxıcı",
    "uğursuz", "mənasız", "dözülməz", "orta", "antitalent",
    "xoşlamadım", "sevmədim", "peşman", "tövsiyə etmirəm", "ikrah",
}

AZ_SUFFIXES = [
    "lərdən", "lardan", "lərə", "lara", "ləri", "ları",
    "lərin", "ların", "lərə", "lara", "lərdə", "larda",
    "lərdə", "larda", "ların", "lerin",
    "dən", "dan", "dır", "dir", "dur", "dür",
    "nın", "nin", "nun", "nün",
    "nda", "ndə", "nda", "ndən",
    "lar", "lər", "ın", "in", "un", "ün",
    "da", "də", "ə", "a", "ı", "i",
    "ır", "ir", "ur", "ür",
    "mış", "miş", "muş", "müş",
    "acaq", "əcək", "malı", "məli",
    "sı", "si", "su", "sü",
]

def lexicon_features(tokens):
    pos = sum(1 for t in tokens if t in POS_WORDS)
    neg = sum(1 for t in tokens if t in NEG_WORDS)
    return pos, neg

def az_stem(word):
    for suffix in AZ_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)]
    return word

def build_vocab(train_data, min_freq=2):
    freq = Counter()
    for tokens, _ in train_data:
        freq.update(az_stem(t) for t in tokens)
    return {w for w, c in freq.items() if c >= min_freq}

def bow_features(tokens, vocab, binary=False):
    stemmed = [az_stem(t) for t in tokens]
    counts = Counter(stemmed)
    if binary:
        return {w: 1 for w in counts if w in vocab}
    return {w: counts[w] for w in counts if w in vocab}

def make_features(tokens, vocab, binary=False):
    feats = bow_features(tokens, vocab, binary)
    pos, neg = lexicon_features(tokens)
    feats["LEX_POS"] = pos
    feats["LEX_NEG"] = neg
    return feats

def split(data, ratio=0.8):
    data = list(data)
    random.shuffle(data)
    k = int(len(data) * ratio)
    return data[:k], data[k:]


# Models

class NaiveBayes:
    def train(self, data):
        self.class_counts = Counter()
        self.word_counts  = defaultdict(Counter)
        self.total_words  = Counter()

        for feats, y in data:
            self.class_counts[y] += 1
            for w, c in feats.items():
                self.word_counts[y][w] += c
                self.total_words[y]    += c

        self.vocab_size = len({w for wc in self.word_counts.values() for w in wc})

    def predict(self, feats):
        scores = {}
        for y in self.class_counts:
            total = sum(self.class_counts.values())
            logp  = math.log(self.class_counts[y] / total)
            for w, c in feats.items():
                num   = self.word_counts[y][w] + 1
                den   = self.total_words[y]    + self.vocab_size
                logp += c * math.log(num / den)
            scores[y] = logp
        return max(scores, key=scores.get)

    def predict_proba(self, feats):
        """Return probability of positive class (label=1)."""
        scores = {}
        for y in self.class_counts:
            total = sum(self.class_counts.values())
            logp  = math.log(self.class_counts[y] / total)
            for w, c in feats.items():
                num   = self.word_counts[y][w] + 1
                den   = self.total_words[y]    + self.vocab_size
                logp += c * math.log(num / den)
            scores[y] = logp
        # Softmax over two classes
        max_score = max(scores.values())
        exp_scores = {y: math.exp(s - max_score) for y, s in scores.items()}
        total_exp = sum(exp_scores.values())
        return exp_scores.get(1, 0) / total_exp


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=10):
        self.lr     = lr
        self.epochs = epochs
        self.w      = defaultdict(float)
        self.bias   = 0.0

    def sigmoid(self, z):
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1 + ez)

    def train(self, data):
        for _ in range(self.epochs):
            for feats, y in data:
                z = sum(self.w[f] * v for f, v in feats.items()) + self.bias
                p = self.sigmoid(z)
                err = y - p
                for f, v in feats.items():
                    self.w[f] += self.lr * err * v
                self.bias += self.lr * err

    def predict(self, feats):
        z = sum(self.w[f] * v for f, v in feats.items()) + self.bias
        return 1 if self.sigmoid(z) >= 0.5 else 0

    def predict_proba(self, feats):
        z = sum(self.w[f] * v for f, v in feats.items()) + self.bias
        return self.sigmoid(z)


# Evaluation

def get_predictions(model, data):
    return [model.predict(f) for f, _ in data]

def confusion_matrix(preds, gold):
    tp = sum(p == 1 and g == 1 for p, g in zip(preds, gold))
    tn = sum(p == 0 and g == 0 for p, g in zip(preds, gold))
    fp = sum(p == 1 and g == 0 for p, g in zip(preds, gold))
    fn = sum(p == 0 and g == 1 for p, g in zip(preds, gold))
    return tp, tn, fp, fn

def precision_recall_f1(preds, gold):
    tp, tn, fp, fn = confusion_matrix(preds, gold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1

def print_metrics(name, preds, gold):
    acc        = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    p, r, f1   = precision_recall_f1(preds, gold)
    tp, tn, fp, fn = confusion_matrix(preds, gold)
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  Confusion matrix  (1=pos / 0=neg):")
    print(f"            Pred+   Pred-")
    print(f"  Actual+   {tp:5d}   {fn:5d}   (TP / FN)")
    print(f"  Actual-   {fp:5d}   {tn:5d}   (FP / TN)")

def mcnemar(pred1, pred2, gold, label1="A", label2="B"):
    b = c = 0
    for p1, p2, y in zip(pred1, pred2, gold):
        if p1 == y and p2 != y:
            b += 1
        elif p1 != y and p2 == y:
            c += 1

    if (b + c) == 0:
        print(f"\nMcNemar ({label1} vs {label2}): b+c=0, models identical.")
        return 0.0

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    sig  = chi2 > 3.841
    print(f"\nMcNemar's test - {label1} vs {label2}:")
    print(f"  b={b}  c={c}  chi²={chi2:.4f}  "
          f"{'Significant (p<0.05)' if sig else 'Not significant (p≥0.05)'}")
    return chi2