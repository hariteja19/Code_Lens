"use client";

import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ── Suggestion questions for the empty state ── */
const SUGGESTIONS = [
  "How does authentication work?",
  "Where is error handling implemented?",
  "Explain the database models",
  "What API endpoints are available?",
  "How is input validation done?",
];

/* ── Tech stack features for the index page ── */
const FEATURES = [
  { icon: "🌳", title: "AST Chunking", desc: "Tree-sitter parses code into semantic units — never splits functions", color: "purple" },
  { icon: "🔍", title: "Hybrid Search", desc: "Dense embeddings + BM25 keywords fused with RRF", color: "blue" },
  { icon: "🎯", title: "Cross-Encoder", desc: "Re-ranking improves faithfulness by 15% over RRF alone", color: "cyan" },
  { icon: "⚡", title: "Groq Inference", desc: "Ultra-fast free LLM inference via Llama 3.3 70B", color: "amber" },
  { icon: "📊", title: "Built-in Eval", desc: "Recall@K and faithfulness metrics with 20 test cases", color: "emerald" },
  { icon: "📎", title: "Exact Citations", desc: "Every answer cites exact file paths and line numbers", color: "rose" },
];

/* ── Tech badges for the footer ── */
const TECH_BADGES = [
  "Tree-sitter", "Qdrant", "nomic-embed", "RRF", "Cross-Encoder",
  "Groq", "FastAPI", "Next.js", "Docker",
];


/* ══════════════════════════════════════════════════════════════
   CodeModal — Full code viewer with line numbers & metadata
   ══════════════════════════════════════════════════════════════ */
function CodeModal({ citation, onClose }) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const handleEsc = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(citation.full_code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
  };

  /* Generate line numbers */
  const lines = citation.full_code.split("\n");
  const lineNumbers = lines.map((_, i) => citation.start_line + i);

  return (
    <div className="code-modal-overlay" onClick={onClose}>
      <div className="code-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="code-modal-header">
          <div>
            <div className="code-modal-title">{citation.file_path}</div>
            <div className="code-modal-subtitle">
              <span>Lines {citation.start_line}–{citation.end_line}</span>
              <span>·</span>
              <span>{citation.node_type.replace(/_/g, " ")}</span>
              {citation.parent_class && (
                <>
                  <span>·</span>
                  <span>in {citation.parent_class}</span>
                </>
              )}
            </div>
          </div>
          <div className="code-modal-actions">
            <button className={`btn btn-sm btn-ghost ${copied ? "copied" : ""}`} onClick={handleCopy}>
              {copied ? "✓ Copied" : "Copy"}
            </button>
            <button className="code-modal-close" onClick={onClose}>✕</button>
          </div>
        </div>

        {/* Metadata chips */}
        <div className="modal-meta">
          <div className="modal-meta-chip">
            <span className="chip-label">Lang</span>
            <span className="chip-value">{citation.language}</span>
          </div>
          <div className="modal-meta-chip">
            <span className="chip-label">Type</span>
            <span className="chip-value">{citation.node_type.replace(/_/g, " ")}</span>
          </div>
          {citation.rerank_score > 0 && (
            <div className="modal-meta-chip">
              <span className="chip-label">Rerank</span>
              <span className="chip-value" style={{ color: "var(--accent-emerald)" }}>
                {citation.rerank_score.toFixed(3)}
              </span>
            </div>
          )}
          {citation.rrf_score > 0 && (
            <div className="modal-meta-chip">
              <span className="chip-label">RRF</span>
              <span className="chip-value">{citation.rrf_score.toFixed(4)}</span>
            </div>
          )}
        </div>

        {/* Code with line numbers */}
        <div className="code-modal-body">
          <div className="code-lines-container">
            <div className="code-line-numbers">
              {lineNumbers.map((num) => (
                <div key={num}>{num}</div>
              ))}
            </div>
            <div className="code-line-content">{citation.full_code}</div>
          </div>
        </div>
      </div>
    </div>
  );
}


/* ══════════════════════════════════════════════════════════════
   CitationCard — Single source reference
   ══════════════════════════════════════════════════════════════ */
function CitationCard({ citation, onClick }) {
  return (
    <div className="citation-card" onClick={() => onClick(citation)}>
      <div className="citation-index">{citation.index}</div>
      <div className="citation-info">
        <div className="citation-file">{citation.file_path}</div>
        <div className="citation-lines">
          Lines {citation.start_line}–{citation.end_line}
          {citation.rerank_score > 0 && (
            <span style={{ marginLeft: 8, color: "var(--accent-emerald)" }}>
              score: {citation.rerank_score.toFixed(2)}
            </span>
          )}
        </div>
        <div className="citation-snippet">{citation.snippet}</div>
      </div>
      <div className="citation-type-badge">
        {citation.node_type.replace(/_/g, " ")}
      </div>
    </div>
  );
}


/* ══════════════════════════════════════════════════════════════
   MessageBubble — A single chat message (user or assistant)
   ══════════════════════════════════════════════════════════════ */
function MessageBubble({ message, onCitationClick }) {
  const isUser = message.role === "user";
  const [copiedIndex, setCopiedIndex] = useState(null);

  const handleCopyCode = async (code, index) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch {}
  };

  /* Enhanced markdown rendering */
  function renderContent(text) {
    if (!text) return null;

    const parts = text.split(/(```[\s\S]*?```|`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\])/g);
    let codeBlockCounter = 0;

    return parts.map((part, i) => {
      /* Fenced code block */
      if (part.startsWith("```")) {
        const langMatch = part.match(/^```(\w+)?\n?/);
        const lang = langMatch?.[1] || "";
        const code = part.replace(/^```\w*\n?/, "").replace(/```$/, "");
        const blockIdx = codeBlockCounter++;
        return (
          <div className="code-block-wrapper" key={i}>
            <button
              className={`copy-btn ${copiedIndex === blockIdx ? "copied" : ""}`}
              onClick={() => handleCopyCode(code, blockIdx)}
            >
              {copiedIndex === blockIdx ? "✓ Copied" : "Copy"}
            </button>
            <pre>
              <code>{code}</code>
            </pre>
          </div>
        );
      }
      /* Inline code */
      if (part.startsWith("`") && part.endsWith("`")) {
        return <code key={i}>{part.slice(1, -1)}</code>;
      }
      /* Bold */
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      /* Citation reference like [1] */
      if (/^\[\d+\]$/.test(part)) {
        return (
          <span
            key={i}
            className="citation-ref"
            style={{
              color: "var(--accent-purple-light)",
              fontWeight: 600,
              cursor: "pointer",
              fontSize: "0.82em",
              background: "rgba(139, 92, 246, 0.12)",
              padding: "1px 5px",
              borderRadius: "4px",
              border: "1px solid rgba(139, 92, 246, 0.2)",
            }}
          >
            {part}
          </span>
        );
      }
      /* Normal text — split on newlines to create paragraphs */
      return part.split("\n").map((line, j) => (
        <span key={`${i}-${j}`}>
          {line}
          {j < part.split("\n").length - 1 && <br />}
        </span>
      ));
    });
  }

  return (
    <div className="message">
      <div className={`message-avatar ${isUser ? "user" : "assistant"}`}>
        {isUser ? "👤" : "🔍"}
      </div>
      <div className="message-body">
        <div className="message-header">
          <span className="message-sender">{isUser ? "You" : "CodeLens"}</span>
          <span className="message-time">{message.time}</span>
        </div>
        <div className="message-content">{renderContent(message.content)}</div>

        {/* Pipeline metadata for assistant messages */}
        {!isUser && message.model && (
          <div className="response-meta">
            <div className="meta-chip">
              <span className="meta-label">Model</span>
              <span className="meta-value">{message.model}</span>
            </div>
            <div className="meta-chip">
              <span className="meta-label">Chunks</span>
              <span className="meta-value">{message.chunks}</span>
            </div>
            {message.provider && (
              <div className="meta-chip">
                <span className="meta-label">Provider</span>
                <span className="meta-value">{message.provider}</span>
              </div>
            )}
            {message.usage && (
              <>
                <div className="meta-chip">
                  <span className="meta-label">In</span>
                  <span className="meta-value">{message.usage.input_tokens?.toLocaleString()}</span>
                </div>
                <div className="meta-chip">
                  <span className="meta-label">Out</span>
                  <span className="meta-value">{message.usage.output_tokens?.toLocaleString()}</span>
                </div>
              </>
            )}
          </div>
        )}

        {/* Citations panel */}
        {message.citations && message.citations.length > 0 && (
          <div className="citations-panel">
            <div className="citations-header">
              <span>📎</span>
              <span>{message.citations.length} Sources Referenced</span>
            </div>
            {message.citations.map((c) => (
              <CitationCard key={c.index} citation={c} onClick={onCitationClick} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}


/* ══════════════════════════════════════════════════════════════
   TypingIndicator — Animated dots while waiting for response
   ══════════════════════════════════════════════════════════════ */
function TypingIndicator() {
  return (
    <div className="message">
      <div className="message-avatar assistant">🔍</div>
      <div className="message-body">
        <div className="message-header">
          <span className="message-sender">CodeLens</span>
          <span className="message-time">searching & generating...</span>
        </div>
        <div className="typing-indicator">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      </div>
    </div>
  );
}


/* ══════════════════════════════════════════════════════════════
   Main Page Component
   ══════════════════════════════════════════════════════════════ */
export default function Home() {
  /* ── State ── */
  const [phase, setPhase] = useState("index"); // "index" | "chat"
  const [repoUrl, setRepoUrl] = useState("");
  const [repoId, setRepoId] = useState("");
  const [indexStatus, setIndexStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [codeModal, setCodeModal] = useState(null);
  const [indexedRepos, setIndexedRepos] = useState([]);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const pollRef = useRef(null);

  /* Fetch previously indexed repos on mount */
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/repos`);
        if (res.ok) {
          const data = await res.json();
          setIndexedRepos(data.repos || []);
        }
      } catch {
        /* backend might not be running yet */
      }
    })();
  }, []);

  /* Auto-scroll to latest message */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  /* Focus input on phase change */
  useEffect(() => {
    if (phase === "chat") {
      setTimeout(() => inputRef.current?.focus(), 300);
    }
  }, [phase]);

  /* Cleanup polling on unmount */
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  /* ── Helper: Get indexing step status ── */
  const getStepState = (stepName) => {
    if (!indexStatus) return "";
    const statusMap = {
      "Clone": { active: ["cloning"], done: ["indexing", "complete"] },
      "Parse": { active: ["indexing"], done: ["complete"] },
      "Embed": { active: ["indexing"], done: ["complete"] },
      "Store": { active: [], done: ["complete"] },
    };
    const step = statusMap[stepName];
    if (!step) return "";
    if (step.done.includes(indexStatus.status)) return "done";
    if (step.active.includes(indexStatus.status)) {
      /* For multi-step "indexing" phase, use progress to differentiate */
      if (stepName === "Parse" && indexStatus.progress < 50) return "active";
      if (stepName === "Embed" && indexStatus.progress >= 50 && indexStatus.progress < 90) return "active";
      if (stepName === "Store" && indexStatus.progress >= 90) return "active";
      if (stepName === "Parse") return "active";
    }
    return "";
  };

  /* ── Index a repository ── */
  const handleIndex = async (e) => {
    e.preventDefault();
    if (!repoUrl.trim()) return;

    setIndexStatus({ status: "queued", progress: 0 });

    try {
      const res = await fetch(`${API_BASE}/api/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_url: repoUrl.trim() }),
      });

      if (!res.ok) throw new Error("Failed to start indexing");

      const data = await res.json();
      setRepoId(data.repo_id);

      /* Poll for status updates */
      pollRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/api/index/${data.job_id}`);
          const statusData = await statusRes.json();
          setIndexStatus(statusData);

          if (statusData.status === "complete" || statusData.status === "error") {
            clearInterval(pollRef.current);
            pollRef.current = null;

            if (statusData.status === "complete") {
              setTimeout(() => setPhase("chat"), 1500);
            }
          }
        } catch {
          /* ignore poll errors */
        }
      }, 2000);
    } catch (err) {
      setIndexStatus({ status: "error", error: err.message });
    }
  };

  /* ── Use a previously indexed repo ── */
  const handleUseRepo = (rid) => {
    setRepoId(rid);
    setPhase("chat");
  };

  /* ── Go back to index phase ── */
  const handleNewRepo = () => {
    setPhase("index");
    setRepoUrl("");
    setRepoId("");
    setIndexStatus(null);
    setMessages([]);
    setInput("");
  };

  /* ── Send a question ── */
  const handleSend = async (questionText) => {
    const question = (questionText || input).trim();
    if (!question || isLoading) return;

    const now = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });

    /* Add user message */
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question, time: now },
    ]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo_id: repoId,
          question: question,
          top_k: 5,
        }),
      });

      if (!res.ok) throw new Error("Query failed");

      const data = await res.json();

      const responseTime = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          citations: data.citations,
          time: responseTime,
          model: data.model_used,
          provider: data.provider,
          chunks: data.chunks_retrieved,
          usage: data.usage,
        },
      ]);
    } catch (err) {
      const errorTime = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Sorry, I encountered an error: ${err.message}. Make sure the backend is running at ${API_BASE} and the repository is indexed.`,
          time: errorTime,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  /* ═════════════════════════════════════════════════════════════
     RENDER
     ═════════════════════════════════════════════════════════════ */
  return (
    <div className="app-layout">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo" onClick={handleNewRepo}>
            <div className="logo-icon">⟨/⟩</div>
            <span>Code<span className="logo-text-gradient">Lens</span></span>
          </div>
          <div className="header-right">
            {phase === "chat" && (
              <>
                <div className="repo-pill">
                  <span>📦</span>
                  <span>{repoId}</span>
                </div>
                <button className="btn btn-sm btn-ghost" onClick={handleNewRepo}>
                  + New Repo
                </button>
              </>
            )}
            <div className="header-badge">
              <span className="header-dot" />
              RAG-Powered
            </div>
          </div>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="main-content">
        {/* ════ INDEX PHASE ════ */}
        {phase === "index" && (
          <div className="phase-enter">
            <div className="glass-card index-section">
              <h1 className="index-title">
                Understand any codebase
                <br />
                <span className="logo-text-gradient">in seconds.</span>
              </h1>
              <p className="index-subtitle">
                Paste a GitHub repository URL and ask questions with natural language.
                Get accurate answers with exact file and line citations.
              </p>

              <form className="index-form" onSubmit={handleIndex}>
                <input
                  id="repo-url-input"
                  type="url"
                  className="index-input"
                  placeholder="https://github.com/user/repo"
                  value={repoUrl}
                  onChange={(e) => setRepoUrl(e.target.value)}
                  disabled={indexStatus && indexStatus.status !== "error"}
                />
                <button
                  id="index-btn"
                  type="submit"
                  className="btn btn-primary"
                  disabled={!repoUrl.trim() || (indexStatus && indexStatus.status !== "error")}
                >
                  {indexStatus && indexStatus.status !== "error" ? (
                    <span className="loading-spinner" />
                  ) : (
                    "Index →"
                  )}
                </button>
              </form>

              {/* Progress with step indicators */}
              {indexStatus && indexStatus.status !== "error" && (
                <div className="progress-container">
                  <div className="progress-steps">
                    {["Clone", "Parse", "Embed", "Store"].map((step) => (
                      <div key={step} className={`progress-step ${getStepState(step)}`}>
                        {getStepState(step) === "done" ? "✓" : getStepState(step) === "active" ? "⟳" : "○"}{" "}
                        {step}
                      </div>
                    ))}
                  </div>
                  <div className="progress-bar-bg">
                    <div
                      className="progress-bar-fill"
                      style={{ width: `${indexStatus.progress || 0}%` }}
                    />
                  </div>
                  <div className="progress-status">
                    <span className="progress-label">
                      {indexStatus.status === "queued" && "⏳ Queued..."}
                      {indexStatus.status === "cloning" && "📥 Cloning repository..."}
                      {indexStatus.status === "indexing" && "🧠 Parsing & embedding chunks..."}
                      {indexStatus.status === "complete" && "✅ Ready!"}
                    </span>
                    <span className="progress-percent">{indexStatus.progress || 0}%</span>
                  </div>
                </div>
              )}

              {/* Success */}
              {indexStatus && indexStatus.status === "complete" && (
                <div className="success-banner">
                  <div className="success-icon">✅</div>
                  <div className="success-text">
                    <div className="success-title">Repository indexed successfully</div>
                    <div className="success-detail">
                      {indexStatus.total_chunks} code chunks extracted and embedded.
                      Redirecting to chat...
                    </div>
                  </div>
                </div>
              )}

              {/* Error */}
              {indexStatus && indexStatus.status === "error" && (
                <div
                  className="success-banner"
                  style={{
                    background: "rgba(244, 63, 94, 0.08)",
                    borderColor: "rgba(244, 63, 94, 0.18)",
                    marginTop: "var(--space-lg)",
                  }}
                >
                  <div className="success-icon">❌</div>
                  <div className="success-text">
                    <div className="success-title" style={{ color: "var(--accent-rose)" }}>
                      Indexing failed
                    </div>
                    <div className="success-detail">{indexStatus.error}</div>
                  </div>
                </div>
              )}

              {/* Tech stack features grid */}
              {!indexStatus && (
                <div className="features-grid">
                  {FEATURES.map((f) => (
                    <div className="feature-card" key={f.title}>
                      <div className={`feature-icon ${f.color}`}>{f.icon}</div>
                      <div>
                        <div className="feature-title">{f.title}</div>
                        <div className="feature-desc">{f.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Previously indexed repos */}
              {!indexStatus && indexedRepos.length > 0 && (
                <div className="indexed-repos">
                  <div className="indexed-repos-title">
                    <span>📚</span>
                    <span>Previously Indexed</span>
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-sm)" }}>
                    {indexedRepos.map((repo) => (
                      <button
                        key={repo.repo_id}
                        className="indexed-repo-chip"
                        onClick={() => handleUseRepo(repo.repo_id)}
                      >
                        ✓ {repo.repo_id}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ════ CHAT PHASE ════ */}
        {phase === "chat" && (
          <div className="glass-card chat-container phase-enter">
            <div className="chat-messages">
              {messages.length === 0 && !isLoading && (
                <div className="empty-state">
                  <div className="empty-icon">🔍</div>
                  <div className="empty-title">Ask anything about {repoId}</div>
                  <div className="empty-description">
                    Ask about functions, architecture, patterns, or any part of the codebase.
                    I&apos;ll retrieve the relevant code and answer with exact citations.
                  </div>
                  <div className="suggestion-chips">
                    {SUGGESTIONS.map((s) => (
                      <button
                        key={s}
                        className="suggestion-chip"
                        onClick={() => handleSend(s)}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <MessageBubble
                  key={i}
                  message={msg}
                  onCitationClick={setCodeModal}
                />
              ))}

              {isLoading && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="chat-input-container">
              <form className="chat-input-form" onSubmit={(e) => { e.preventDefault(); handleSend(); }}>
                <div className="chat-input-wrapper">
                  <textarea
                    ref={inputRef}
                    id="chat-input"
                    className="chat-input"
                    placeholder={`Ask about ${repoId}...`}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={1}
                    disabled={isLoading}
                  />
                </div>
                <button
                  id="send-btn"
                  type="submit"
                  className="btn btn-primary btn-send"
                  disabled={!input.trim() || isLoading}
                >
                  ↑
                </button>
              </form>
              <div className="chat-input-hint">
                <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for new line
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        <div className="tech-badges">
          {TECH_BADGES.map((badge) => (
            <span key={badge} className="tech-badge">{badge}</span>
          ))}
        </div>
        <div className="footer-text">
          Built by HariTeja · AST-aware chunking, hybrid search with RRF, and cross-encoder re-ranking
        </div>
      </footer>

      {/* ── Code Viewer Modal ── */}
      {codeModal && (
        <CodeModal citation={codeModal} onClose={() => setCodeModal(null)} />
      )}
    </div>
  );
}
