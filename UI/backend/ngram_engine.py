import math
from collections import Counter, defaultdict

import nltk
import numpy as np


# Corpus

def _load_fuzuli(data_dir: str) -> str:
    import os
    text = " "
    for i in range(1, 6):
        with open(os.path.join(data_dir, f"fuzuli-{i}.md"), "r", encoding="utf-8") as f:
            text += f.read() + " "
    return text

def _load_hatai(data_dir: str) -> str:
    import os
    with open(os.path.join(data_dir, "hatai-1.md"), "r", encoding="utf-8") as f:
        return f.read()


# Model

class NgramModel:
    """Mirrors ngram_actual.ipynb cell by cell."""

    def __init__(self, data_dir: str):

        raw_tokens = _load_fuzuli(data_dir).split()
        freq_temp  = nltk.FreqDist(raw_tokens)
        rare       = {w for w in raw_tokens if freq_temp[w] == 1}
        self.tokens       = [w if w not in rare else "UNK" for w in raw_tokens]
        self.freq_dist    = nltk.FreqDist(self.tokens)
        self.vocab        = list(self.freq_dist.keys())
        self.V            = len(self.vocab)
        self.N            = len(self.tokens)
        self.word_to_idx  = {word: i for i, word in enumerate(self.vocab)}

        self.bigrams      = [(self.tokens[i], self.tokens[i+1])
                             for i in range(self.N - 1)]
        self.bigram_count = Counter(self.bigrams)

        self.trigrams      = [(self.tokens[i], self.tokens[i+1], self.tokens[i+2])
                              for i in range(self.N - 2)]
        self.trigram_count = Counter(self.trigrams)

        self.bigram_to_idx = {(w1, w2): i
                              for i, (w1, w2) in enumerate(self.bigram_count.keys())}

        self._em_lambdas()

        self._kn_precompute()

        raw_test   = _load_hatai(data_dir).split()
        self.test  = [w if w in self.vocab else "UNK" for w in raw_test]

    # M

    def _em_lambdas(self):
        l1, l2 = 0.5, 0.5
        for _ in range(20):
            g1 = g2 = 0.0
            for i in range(1, self.N):
                w1, w2 = self.tokens[i-1], self.tokens[i]
                n1 = l1 * self._p_uni(w2)
                n2 = l2 * self._p_bi(w1, w2)
                d  = n1 + n2
                if d:
                    g1 += n1/d; g2 += n2/d
            t = g1 + g2
            l1, l2 = g1/t, g2/t
        self.lam_bi = (l1, l2)

        l1, l2, l3 = 1/3, 1/3, 1/3
        for _ in range(30):
            g1 = g2 = g3 = 0.0
            for i in range(2, self.N):
                w1, w2, w3 = self.tokens[i-2], self.tokens[i-1], self.tokens[i]
                n1 = l1 * self._p_uni(w3)
                n2 = l2 * self._p_bi(w2, w3)
                n3 = l3 * self._p_tri(w1, w2, w3)
                d  = n1 + n2 + n3
                if d:
                    g1 += n1/d; g2 += n2/d; g3 += n3/d
            t = g1 + g2 + g3
            l1, l2, l3 = g1/t, g2/t, g3/t
        self.lam_tri = (l1, l2, l3)

    # KN precompute

    def _kn_precompute(self):
        self.D = 0.75
        continuation_counts = defaultdict(set)
        for (w1, w2) in self.bigram_count:
            continuation_counts[w2].add(w1)
        total_bigram_types = len(self.bigram_count)
        self.continuation_probs = {
            w2: len(continuation_counts[w2]) / total_bigram_types
            for w2 in self.vocab
        }
        self.c_w1_dict             = defaultdict(int)
        self.num_bigrams_from_w1   = defaultdict(int)
        for (w1, w2), count in self.bigram_count.items():
            self.c_w1_dict[w1]           += count
            self.num_bigrams_from_w1[w1] += 1
        self.num_trigrams_from_w1w2 = defaultdict(int)
        for (w1, w2, w3) in self.trigram_count:
            self.num_trigrams_from_w1w2[(w1, w2)] += 1

    # Raw MLE

    def _p_uni(self, w):
        return self.freq_dist[w] / self.N

    def _p_bi(self, w1, w2):
        d = self.freq_dist[w1]
        return self.bigram_count[(w1, w2)] / d if d else 0.0

    def _p_tri(self, w1, w2, w3):
        d = self.bigram_count[(w1, w2)]
        return self.trigram_count[(w1, w2, w3)] / d if d else 0.0

    # Build probability arrays for a given mode

    def _build_unigram(self, mode: str, k: float = 0.5) -> np.ndarray:
        arr = np.zeros(self.V)
        for i, word in enumerate(self.vocab):
            c = self.freq_dist[word]
            if   mode == "none":    arr[i] = c / self.N
            elif mode == "laplace": arr[i] = (c+1) / (self.N + self.V)
            elif mode == "k":       arr[i] = (c+k) / (self.N + k*self.V)
            else:                   arr[i] = c / self.N
        return arr

    def _build_bigram(self, mode: str, unigram: np.ndarray,
                      k: float = 0.5, lambda_bo: float = 0.4) -> np.ndarray:
        """Fills only observed bigrams — matches Cell 7."""
        mat = np.zeros((self.V, self.V))
        for (w1, w2) in self.bigrams:
            i     = self.word_to_idx[w1]
            j     = self.word_to_idx[w2]
            count = self.bigram_count[(w1, w2)]
            d     = self.freq_dist[w1]

            if mode == "none":
                mat[i, j] = count / d if d else 0.0
            elif mode == "laplace":
                mat[i, j] = (count+1) / (d + self.V) if d else 1/self.V
            elif mode == "k":
                mat[i, j] = (count+k) / (d + k*self.V) if d else 0.0
            elif mode == "interpolation":
                l1, l2 = self.lam_bi
                mat[i, j] = l1*self._p_uni(w2) + l2*self._p_bi(w1, w2)
            elif mode == "backoff":
                mat[i, j] = count / d if d else lambda_bo * unigram[j]
            elif mode == "kneser_ney":
                c_w1   = self.c_w1_dict[w1]
                lam_w1 = (self.D * self.num_bigrams_from_w1[w1] / c_w1
                          if c_w1 > 0 else 1.0)
                P_cont = self.continuation_probs.get(w2, 0.0)
                mat[i, j] = max(count - self.D, 0)/c_w1 + lam_w1*P_cont if c_w1 else P_cont
        return mat

    def _build_trigram(self, mode: str, unigram: np.ndarray,
                       bigram_mat: np.ndarray,
                       k: float = 0.5, lambda_bo: float = 0.4) -> np.ndarray:
        n_bi = len(self.bigram_count)
        mat  = np.zeros((n_bi, self.V))
        for (w1, w2, w3) in self.trigrams:
            if (w1, w2) not in self.bigram_to_idx:
                continue
            i     = self.bigram_to_idx[(w1, w2)]
            j     = self.word_to_idx[w3]
            count = self.trigram_count[(w1, w2, w3)]
            denom = self.bigram_count.get((w1, w2), 0)

            if mode == "none":
                mat[i, j] = count / denom if denom else 0.0
            elif mode == "laplace":
                mat[i, j] = (count+1) / (denom + self.V) if denom else 1/self.V
            elif mode == "k":
                mat[i, j] = (count+k) / (denom + k*self.V) if denom else 0.0
            elif mode == "interpolation":
                l1, l2, l3 = self.lam_tri
                mat[i, j] = (l1*self._p_uni(w3)
                             + l2*self._p_bi(w2, w3)
                             + l3*self._p_tri(w1, w2, w3))
            elif mode == "backoff":
                if count > 0:
                    mat[i, j] = count / denom
                elif self.bigram_count.get((w2, w3), 0) > 0:
                    mat[i, j] = lambda_bo * bigram_mat[self.word_to_idx[w2],
                                                       self.word_to_idx[w3]]
                else:
                    mat[i, j] = lambda_bo**2 * unigram[j]
            elif mode == "kneser_ney":
                c_w1w2 = denom
                if c_w1w2 > 0:
                    lam_w1w2 = (self.D * self.num_trigrams_from_w1w2.get((w1,w2), 0)
                                / c_w1w2)
                else:
                    lam_w1w2 = 1.0
                P_back = bigram_mat[self.word_to_idx[w2], self.word_to_idx[w3]]
                mat[i, j] = (max(count - self.D, 0)/c_w1w2 + lam_w1w2*P_back
                             if c_w1w2 else P_back)
        return mat

    # Perplexity

    def perplexity_unigram(self, mode: str) -> float:
        uni   = self._build_unigram(mode)
        log_p = 0.0
        for w in self.test:
            p = uni[self.word_to_idx[w]]
            log_p += math.log(p) if p > 0 else -1e9
        return math.exp(-log_p / len(self.test))

    def perplexity_bigram(self, mode: str) -> float:
        uni = self._build_unigram(mode)
        bi  = self._build_bigram(mode, uni)
        pairs = [(self.test[i], self.test[i+1]) for i in range(len(self.test)-1)]
        pairs = [(w1, w2) if (w1,w2) in self.bigram_count else ("UNK","UNK")
                 for w1, w2 in pairs]
        log_p = 0.0
        for w1, w2 in pairs:
            p = bi[self.word_to_idx[w1], self.word_to_idx[w2]]
            log_p += math.log(p) if p > 0 else -1e9
        return math.exp(-log_p / len(pairs))

    def perplexity_trigram(self, mode: str) -> float:
        uni = self._build_unigram(mode)
        bi  = self._build_bigram(mode, uni)
        tri = self._build_trigram(mode, uni, bi)
        triples = [(self.test[i], self.test[i+1], self.test[i+2])
                   for i in range(len(self.test)-2)]
        triples = [(w1,w2,w3) if (w1,w2,w3) in self.trigram_count else ("UNK","UNK","UNK")
                   for w1,w2,w3 in triples]
        unk_bi  = self.bigram_to_idx.get(("UNK","UNK"), 0)
        log_p   = 0.0
        for w1, w2, w3 in triples:
            bi_i = self.bigram_to_idx.get((w1,w2), unk_bi)
            p    = tri[bi_i, self.word_to_idx[w3]]
            log_p += math.log(p) if p > 0 else -1e9
        return math.exp(-log_p / len(triples))

    def stats(self) -> dict:
        return {
            "n_tokens":   self.N,
            "vocab_size": self.V,
            "n_unigrams": self.V,
            "n_bigrams":  len(self.bigram_count),
            "n_trigrams": len(self.trigram_count),
        }