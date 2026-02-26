import { useState, useEffect, useRef } from "react";
import "./css/SentimentPage.css";

const API = "http://localhost:8000";

const MODEL_META = {
  nb:  { name: "Naïve Bayes",         short: "NB",  desc: "Count bag-of-words" },
  bnb: { name: "Binary Naïve Bayes",  short: "BNB", desc: "Binary bag-of-words" },
  lr:  { name: "Logistic Regression", short: "LR",  desc: "Count bag-of-words" },
};

const METRIC_LABELS = {
  accuracy:  "Accuracy",
  precision: "Precision",
  recall:    "Recall",
  f1:        "F1 Score",
};

const pct = (v) => `${(v * 100).toFixed(1)}%`;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ── Sub-components ────────────────────────────────────────────────────────────

function GaugeMini({ value, positive }) {
  const r = 32, cx = 40, cy = 42, stroke = 7;
  const circ = Math.PI * r;
  const fill = clamp(value, 0, 1) * circ;
  const color = positive ? "#4ade80" : "#f87171";
  return (
    <svg width="80" height="50" viewBox="0 0 80 50" style={{ overflow: "visible" }}>
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke="#1e293b" strokeWidth={stroke}
      />
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeDasharray={`${fill} ${circ}`}
        strokeLinecap="round"
        style={{ transition: "stroke-dasharray 0.6s cubic-bezier(.4,0,.2,1)" }}
      />
      <text x={cx} y={cy - 6} textAnchor="middle" fontSize="11"
            fill={color} fontFamily="'JetBrains Mono', monospace" fontWeight="700">
        {pct(value)}
      </text>
    </svg>
  );
}

function MetricBar({ value }) {
  return (
    <div className="metric-bar-wrap">
      <div className="metric-bar-track">
        <div className="metric-bar-fill" style={{ width: pct(value) }} />
      </div>
      <span className="metric-bar-label">{pct(value)}</span>
    </div>
  );
}

function PredictionCard({ model, result, loading }) {
  const meta = MODEL_META[model];
  const isPos = result?.label === "positive";
  const conf  = result?.confidence ?? 0;
  const accentColor = result ? (isPos ? "#4ade80" : "#f87171") : "#334155";

  return (
    <div
      className="prediction-card"
      style={{ borderColor: result ? `${accentColor}40` : "#1e293b" }}
    >
      {result && (
        <div
          className="prediction-card-glow"
          style={{ background: accentColor }}
        />
      )}

      <div className="prediction-card-header">
        <div>
          <span className="model-short-badge">{meta.short}</span>
          <div className="model-name">{meta.name}</div>
          <div className="model-desc">{meta.desc}</div>
        </div>
        {result && (
          <span
            className={`sentiment-badge ${isPos ? "sentiment-pos" : "sentiment-neg"}`}
          >
            {result.label}
          </span>
        )}
      </div>

      {loading ? (
        <div className="loading-dots">
          {[0, 1, 2].map(i => (
            <div key={i} className="dot" style={{ animationDelay: `${i * 0.2}s` }} />
          ))}
        </div>
      ) : result ? (
        <div className="gauge-wrap">
          <GaugeMini value={conf} positive={isPos} />
        </div>
      ) : (
        <div className="awaiting">awaiting input…</div>
      )}
    </div>
  );
}

function MetricsPanel({ metrics }) {
  if (!metrics) return null;
  return (
    <div className="metrics-panel">
      <div className="metrics-header">
        <span className="metrics-title">Model Evaluation</span>
        <div className="metrics-divider" />
        <span className="metrics-meta">
          {metrics.train_size?.toLocaleString()} train ·{" "}
          {metrics.test_size?.toLocaleString()} test ·{" "}
          {metrics.vocab_size?.toLocaleString()} vocab
        </span>
      </div>

      <div className="metrics-grid">
        {Object.entries(MODEL_META).map(([key, meta]) => {
          const m = metrics[key];
          if (!m) return null;
          return (
            <div key={key} className="metrics-model-col">
              <div className="metrics-model-name">
                <span className="model-short-badge accent">{meta.short}</span>
                {meta.name}
              </div>
              {Object.entries(METRIC_LABELS).map(([mk, ml]) => (
                <div key={mk} className="metric-row">
                  <div className="metric-label">{ml}</div>
                  <MetricBar value={m[mk]} />
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function VerdictBanner({ results }) {
  if (!results) return null;
  const labels    = Object.values(results).map(r => r.label);
  const posCount  = labels.filter(l => l === "positive").length;
  const majority  = posCount > labels.length / 2 ? "positive" : "negative";
  const isPos     = majority === "positive";

  return (
    <div className={`verdict-banner ${isPos ? "verdict-pos" : "verdict-neg"}`}>
      <div>
        <div className={`verdict-subtitle ${isPos ? "verdict-subtitle-pos" : "verdict-subtitle-neg"}`}>
          Summary
        </div>
        <div className="verdict-text">
          {posCount} of {labels.length} models predict{" "}
          <span className={isPos ? "color-pos" : "color-neg"}>{majority}</span>
        </div>
      </div>
    </div>
  );
}

// ── Fuzuli N-gram Panel ───────────────────────────────────────────────────────

const SMOOTHING_OPTIONS = [
  { value: "none",          label: "Unsmoothed",      desc: "Raw MLE — zero probability for unseen n-grams" },
  { value: "laplace",       label: "Laplace (Add-1)", desc: "Add 1 to every count" },
  { value: "k",             label: "Add-k (k=0.5)",   desc: "Add 0.5 to every count" },
  { value: "interpolation", label: "Interpolation",   desc: "EM-estimated weighted mix of unigram, bigram, and trigram" },
  { value: "backoff",       label: "Stupid Backoff",  desc: "Back off to lower-order model when count is zero (λ=0.4)" },
  { value: "kneser_ney",    label: "Kneser-Ney",      desc: "Continuation probability with absolute discount D=0.75" },
];

const fmt = (v) =>
  v === null || v === undefined
    ? "∞"
    : v > 999999
    ? v.toExponential(2)
    : v.toLocaleString(undefined, { maximumFractionDigits: 1 });

function FuzuliPanel() {
  const [stats, setStats]           = useState(null);
  const [mode, setMode]             = useState("laplace");
  const [perplexity, setPerplexity] = useState(null);
  const [ppxLoading, setPpxLoading] = useState(false);
  const [statsError, setStatsError] = useState(false);

  useEffect(() => {
    fetch(`${API}/ngram/stats`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(setStats)
      .catch(() => setStatsError(true));
  }, []);

  useEffect(() => {
    if (!stats) return;
    setPpxLoading(true);
    setPerplexity(null);
    fetch(`${API}/ngram/perplexity?mode=${mode}`)
      .then(r => r.json())
      .then(data => { setPerplexity(data); setPpxLoading(false); })
      .catch(() => setPpxLoading(false));
  }, [mode, stats]);

  const selectedOpt = SMOOTHING_OPTIONS.find(o => o.value === mode);

  return (
    <div className="fuzuli-panel">
      {/* Header — mirrors MetricsPanel */}
      <div className="fuzuli-panel-header">
        <span className="fuzuli-panel-title">Fuzuli · N-gram Language Model</span>
        <div className="fuzuli-panel-divider" />
      </div>

      <div className="fuzuli-top">
        <div className="fuzuli-portrait-wrap">
          <img
            src="https://alchetron.com/cdn/fuzl-8c03be1c-c67f-40a8-be0c-87a86f962e5-resize-750.jpeg"
            alt="Fuzuli"
            className="fuzuli-portrait"
          />
          <div className="fuzuli-name-tag">Fuzuli</div>
        </div>

        <div className="fuzuli-corpus-info">
          <div className="fuzuli-corpus-title">
            Fuzuli Corpus
            <span className="fuzuli-books-badge">6 books</span>
          </div>
          <p className="fuzuli-corpus-desc">
            A collection of six literary works by the Azerbaijani poet Muhammad ibn Suleyman Fuzuli
            (c. 1494–1556), used here to build and evaluate n-gram language models.
          </p>

          {statsError && (
            <div className="fuzuli-stat-error">
              Corpus stats unavailable — is the backend running with FUZULI_CORPUS_DIR set?
            </div>
          )}

          {stats && (
            <div className="fuzuli-stats-grid">
              {[
                { val: stats.n_tokens,   key: "tokens"   },
                { val: stats.vocab_size, key: "vocab"    },
                { val: stats.n_unigrams, key: "unigrams" },
                { val: stats.n_bigrams,  key: "bigrams"  },
                { val: stats.n_trigrams, key: "trigrams" },
              ].map(({ val, key }) => (
                <div key={key} className="fuzuli-stat">
                  <span className="fuzuli-stat-val">{val.toLocaleString()}</span>
                  <span className="fuzuli-stat-key">{key}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="fuzuli-smoothing-section">
        <div className="fuzuli-section-label">SMOOTHING METHOD</div>
        <div className="fuzuli-smoothing-pills">
          {SMOOTHING_OPTIONS.map(opt => (
            <button
              key={opt.value}
              className={`smoothing-pill ${mode === opt.value ? "smoothing-pill-active" : ""}`}
              onClick={() => setMode(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
        {selectedOpt && (
          <div className="fuzuli-smoothing-desc">{selectedOpt.desc}</div>
        )}
      </div>

      <div className="fuzuli-ppx-section">
        <div className="fuzuli-section-label">PERPLEXITY — TEST CORPUS (HATAI)</div>
        <div className="fuzuli-ppx-grid">
          {["unigram", "bigram", "trigram"].map(order => (
            <div key={order} className="fuzuli-ppx-card">
              <div className="fuzuli-ppx-order">{order}</div>
              {ppxLoading ? (
                <div className="loading-dots" style={{ justifyContent: "center", padding: "12px 0" }}>
                  {[0,1,2].map(i => (
                    <div key={i} className="dot" style={{ animationDelay: `${i*0.2}s` }} />
                  ))}
                </div>
              ) : (
                <div className="fuzuli-ppx-val">
                  {perplexity ? fmt(perplexity[order]) : "—"}
                </div>
              )}
            </div>
          ))}
        </div>
        {perplexity && (
          <div className="fuzuli-ppx-note">
            Lower perplexity = better model fit. Evaluated on Hatai as an out-of-domain test set.
          </div>
        )}
      </div>
    </div>
  );
}


// ── Main Page ─────────────────────────────────────────────────────────────────

export default function SentimentPage() {
  const [text, setText]             = useState("");
  const [results, setResults]       = useState(null);
  const [loading, setLoading]       = useState(false);
  const [metrics, setMetrics]       = useState(null);
  const [apiError, setApiError]     = useState(null);
  const [tokensFound, setTokensFound] = useState(null);
  const textRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/metrics`)
      .then(r => r.json())
      .then(setMetrics)
      .catch(() => {});
  }, []);

  async function handleAnalyze() {
    if (!text.trim()) return;
    setLoading(true);
    setResults(null);
    setApiError(null);
    setTokensFound(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "API error");
      }
      const data = await res.json();
      const { tokens_found, ...modelResults } = data;
      setResults(modelResults);
      setTokensFound(tokens_found);
    } catch (e) {
      setApiError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") handleAnalyze();
  }

  const charCount = text.length;
  const overLimit = charCount > 2000;

  return (
    <div className="page-shell">
      <div className="page-content">

        {/* Header */}
        <div className="header">
          {/*<div className="status-pill">
            <span className="status-dot" />
            <span className="status-text">AZ-SENTIMENT · v1.0</span>
          </div>*/}
          <h1 className="hero-title">
            Azerbaijani Sentiment<br />Analysis
          </h1>
          <p className="hero-subtitle">
            Three models analyse the sentiment of Azerbaijani text.
            Enter a review below and compare predictions across Naïve Bayes,
            Binary Naïve Bayes, and Logistic Regression.
          </p>
        </div>

        {/* Input card */}
        <div className="input-card">
          <div className="textarea-wrap">
            <textarea
              ref={textRef}
              rows={5}
              value={text}
              onChange={e => setText(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Bu film həqiqətən çox yaxşı idi…"
              maxLength={2000}
            />
          </div>
          <div className="input-footer">
            <span className={`char-count ${overLimit ? "char-over" : ""}`}>
              {charCount} / 2000
            </span>
            <div className="input-actions">
              {text && (
                <button
                  className="btn-clear"
                  onClick={() => { setText(""); setResults(null); setApiError(null); setTokensFound(null); }}
                >
                  Clear
                </button>
              )}
              <button
                className="btn-analyse"
                onClick={handleAnalyze}
                disabled={!text.trim() || loading || overLimit}
              >
                {loading ? (
                  <>
                    <span className="spinner" />
                    Analysing
                  </>
                ) : "Analyse →"}
              </button>
            </div>
          </div>
        </div>

        {/* API error */}
        {apiError && (
          <div className="error-banner fade-in">
            ⚠ {apiError} — Is the backend running on port 8000?
          </div>
        )}

        {/* Token count */}
        {tokensFound !== null && (
          <div className="token-info">
            <span className="token-count">{tokensFound}</span>{" "}
            token{tokensFound !== 1 ? "s" : ""} matched vocabulary after stemming &amp; stopword removal
          </div>
        )}

        {/* Prediction cards */}
        {(results || loading) && (
          <div className="cards-row fade-in">
            {Object.keys(MODEL_META).map(key => (
              <PredictionCard
                key={key}
                model={key}
                result={results?.[key] ?? null}
                loading={loading}
              />
            ))}
          </div>
        )}

        {/* Summary */}
        {results && (
          <div className="fade-in">
            <VerdictBanner results={results} />
          </div>
        )}

        {/* Sample texts */}
        {!results && !loading && (
          <div className="samples-wrap">
            <div className="samples-label">TRY A SAMPLE</div>
            <div className="samples-list">
              {[
                "Bu film həqiqətən çox yaxşı idi, möhtəşəm rejissor işi.",
                "Dəhşətli film, vaxtımı boşa xərclədim. Tövsiyə etmirəm.",
                "Maraqlı süjet, amma aktyor oyunu zəif idi.",
              ].map((s, i) => (
                <button key={i} className="sample-btn" onClick={() => setText(s)}>
                  "{s.substring(0, 55)}…"
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Metrics panel */}
        {metrics ? (
          <MetricsPanel metrics={metrics} />
        ) : (
          <div className="metrics-placeholder">
            Model evaluation metrics will appear here once the backend is running.
          </div>
        )}

        {/* Fuzuli N-gram panel */}
        <FuzuliPanel />

        {/* Footer */}
        <div className="page-footer">
          <span>Naive Bayes · Binary NB · Logistic Regression</span>
          <span>
            Dataset:{" "}
            <a href="https://github.com/AzTextCorpus/az-sentiment-analysis-dataset" target="_blank" rel="noreferrer">
              AzTextCorpus
            </a>
          </span>
          
        </div>

      </div>
    </div>
  );
}