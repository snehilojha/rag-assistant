import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_URL = "http://localhost:8000/ask";
const BOOKS_URL = "http://localhost:8000/books";

function BookSelector({ books, selected, onChange }) {
  /**
   * Pill/chip selector for filtering which books to query.
   * selected: array of slugs. Empty array = all books.
   * Clicking a chip toggles it. "All" chip clears selection.
   */
  const allActive = selected.length === 0;

  const toggle = (slug) => {
    if (selected.includes(slug)) {
      onChange(selected.filter((s) => s !== slug));
    } else {
      onChange([...selected, slug]);
    }
  };

  const chipBase = {
    fontFamily: "'Space Mono', monospace",
    fontSize: "10px",
    letterSpacing: "0.8px",
    textTransform: "uppercase",
    padding: "5px 12px",
    borderRadius: "20px",
    border: "1px solid",
    cursor: "pointer",
    background: "none",
    transition: "all 0.18s",
    whiteSpace: "nowrap",
  };

  return (
    <div
      style={{
        display: "flex",
        gap: "8px",
        flexWrap: "wrap",
        alignItems: "center",
        marginBottom: "10px",
      }}
    >
      <span
        style={{
          fontFamily: "'Space Mono', monospace",
          fontSize: "9px",
          color: "#484f58",
          letterSpacing: "1.5px",
          textTransform: "uppercase",
          flexShrink: 0,
        }}
      >
        Sources:
      </span>
      {/* All chip */}
      <button
        onClick={() => onChange([])}
        style={{
          ...chipBase,
          borderColor: allActive ? "#64ffda" : "#30363d",
          color: allActive ? "#64ffda" : "#484f58",
          background: allActive ? "rgba(100,255,218,0.07)" : "none",
        }}
      >
        All
      </button>
      {books.map((book) => {
        const active = selected.includes(book.slug);
        return (
          <button
            key={book.slug}
            onClick={() => toggle(book.slug)}
            title={`${book.title} — ${book.author}`}
            style={{
              ...chipBase,
              borderColor: active ? "#64ffda" : "#30363d",
              color: active ? "#64ffda" : "#484f58",
              background: active ? "rgba(100,255,218,0.07)" : "none",
            }}
          >
            {book.title}
          </button>
        );
      })}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{ display: "flex", gap: "5px", padding: "12px 0" }}>
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          style={{
            width: "7px",
            height: "7px",
            borderRadius: "50%",
            background: "#64ffda",
            animation: "bounce 1.2s infinite",
            animationDelay: `${i * 0.2}s`,
          }}
        />
      ))}
    </div>
  );
}

function ChunkDrawer({ chunks, open, onClose }) {
  return (
    <div
      style={{
        position: "fixed",
        right: open ? 0 : "-420px",
        top: 0,
        width: "400px",
        height: "100vh",
        background: "#0d1117",
        borderLeft: "1px solid #21262d",
        transition: "right 0.35s cubic-bezier(0.4,0,0.2,1)",
        zIndex: 100,
        display: "flex",
        flexDirection: "column",
        boxShadow: open ? "-8px 0 32px rgba(0,0,0,0.5)" : "none",
      }}
    >
      <div
        style={{
          padding: "20px 24px",
          borderBottom: "1px solid #21262d",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span style={{ fontFamily: "'Space Mono', monospace", fontSize: "11px", color: "#64ffda", letterSpacing: "2px", textTransform: "uppercase" }}>
          Retrieved Chunks ({chunks.length})
        </span>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            color: "#8b949e",
            cursor: "pointer",
            fontSize: "18px",
            lineHeight: 1,
            padding: "2px 6px",
          }}
        >
          ✕
        </button>
      </div>
      <div style={{ overflowY: "auto", flex: 1, padding: "16px 24px", display: "flex", flexDirection: "column", gap: "16px" }}>
        {chunks.map((chunk, i) => {
          const text = typeof chunk === "object" ? chunk.text : chunk;
          const book = typeof chunk === "object" ? chunk.book : null;
          const page = typeof chunk === "object" ? chunk.page : null;
          return (
            <div
              key={i}
              style={{
                background: "#161b22",
                border: "1px solid #21262d",
                borderRadius: "8px",
                padding: "14px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "8px" }}>
                <span style={{ fontFamily: "'Space Mono', monospace", fontSize: "10px", color: "#64ffda", opacity: 0.7 }}>
                  CHUNK {i + 1}
                </span>
                {(book || page != null) && (
                  <span style={{ fontFamily: "'Space Mono', monospace", fontSize: "9px", color: "#484f58", letterSpacing: "0.5px" }}>
                    {book && <span>{book}</span>}{book && page != null && " · "}{page != null && <span>p. {page}</span>}
                  </span>
                )}
              </div>
              <p style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11px", color: "#8b949e", lineHeight: 1.7, margin: 0 }}>
                {text}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Message({ msg, onShowChunks }) {
  const isUser = msg.role === "user";
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        gap: "6px",
        animation: "fadeSlideIn 0.3s ease forwards",
      }}
    >
      <div
        style={{
          fontFamily: "'Space Mono', monospace",
          fontSize: "9px",
          color: "#484f58",
          letterSpacing: "1px",
          textTransform: "uppercase",
          paddingLeft: isUser ? 0 : "4px",
          paddingRight: isUser ? "4px" : 0,
        }}
      >
        {isUser ? "You" : "DS Tutor"}
      </div>
      <div
        style={{
          maxWidth: "80%",
          background: isUser ? "#1f3a2d" : "#161b22",
          border: `1px solid ${isUser ? "#2ea043" : "#21262d"}`,
          borderRadius: isUser ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
          padding: "14px 18px",
        }}
      >
        {isUser ? (
          <p
            style={{
              fontFamily: "'IBM Plex Sans', sans-serif",
              fontSize: "14px",
              color: "#aff5c0",
              lineHeight: 1.75,
              margin: 0,
              whiteSpace: "pre-wrap",
            }}
          >
            {msg.content}
          </p>
        ) : (
          <div className="md-prose">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {msg.content}
            </ReactMarkdown>
          </div>
        )}
      </div>
      {!isUser && msg.chunks && msg.chunks.length > 0 && (
        <button
          onClick={() => onShowChunks(msg.chunks)}
          style={{
            background: "none",
            border: "1px solid #21262d",
            borderRadius: "6px",
            color: "#484f58",
            fontFamily: "'Space Mono', monospace",
            fontSize: "9px",
            letterSpacing: "1px",
            textTransform: "uppercase",
            padding: "4px 10px",
            cursor: "pointer",
            transition: "all 0.2s",
          }}
          onMouseEnter={(e) => { e.target.style.color = "#64ffda"; e.target.style.borderColor = "#64ffda33"; }}
          onMouseLeave={(e) => { e.target.style.color = "#484f58"; e.target.style.borderColor = "#21262d"; }}
        >
          View {msg.chunks.length} source chunks →
        </button>
      )}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Ask me anything from the indexed books. Select a source chip to narrow your query, or leave All active to search across everything.",
      chunks: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [drawerChunks, setDrawerChunks] = useState([]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [books, setBooks] = useState([]);
  const [selectedBooks, setSelectedBooks] = useState([]);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    fetch(BOOKS_URL)
      .then((r) => r.json())
      .then((data) => setBooks(data))
      .catch(() => {
        /* silently degrade — chips just won't appear */
      });
  }, []);

  const handleShowChunks = (chunks) => {
    setDrawerChunks(chunks);
    setDrawerOpen(true);
  };

  const handleSubmit = async () => {
    const query = input.trim();
    if (!query || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: query,
          sources: selectedBooks.length > 0 ? selectedBooks : null,
        }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, chunks: data.chunks },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Failed to reach the backend. Make sure the FastAPI server is running on port 8000.", chunks: [] },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #010409; }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-6px); opacity: 1; }
        }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #21262d; border-radius: 2px; }
        textarea:focus { outline: none; }
        textarea { resize: none; }
        .md-prose { font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; color: #c9d1d9; line-height: 1.75; }
        .md-prose p { margin: 0 0 10px 0; }
        .md-prose p:last-child { margin-bottom: 0; }
        .md-prose h1, .md-prose h2, .md-prose h3 { font-family: 'Space Mono', monospace; color: #64ffda; margin: 16px 0 8px; font-size: 13px; letter-spacing: 0.5px; }
        .md-prose ul, .md-prose ol { padding-left: 20px; margin: 6px 0 10px; }
        .md-prose li { margin-bottom: 4px; }
        .md-prose code { font-family: 'IBM Plex Mono', monospace; font-size: 12px; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 1px 5px; color: #79c0ff; }
        .md-prose pre { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 14px; overflow-x: auto; margin: 10px 0; }
        .md-prose pre code { background: none; border: none; padding: 0; color: #c9d1d9; font-size: 12px; }
        .md-prose strong { color: #e6edf3; font-weight: 600; }
        .md-prose em { color: #8b949e; }
        .md-prose blockquote { border-left: 3px solid #30363d; padding-left: 12px; margin: 8px 0; color: #8b949e; }
      `}</style>

      {/* Header */}
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          height: "56px",
          background: "rgba(1,4,9,0.85)",
          backdropFilter: "blur(12px)",
          borderBottom: "1px solid #21262d",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 32px",
          zIndex: 50,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              background: "#64ffda",
              animation: "pulse 2s infinite",
            }}
          />
          <span
            style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: "12px",
              color: "#c9d1d9",
              letterSpacing: "1px",
            }}
          >
            DS<span style={{ color: "#64ffda" }}>://</span>scratch
          </span>
        </div>
        <span
          style={{
            fontFamily: "'Space Mono', monospace",
            fontSize: "9px",
            color: "#484f58",
            letterSpacing: "2px",
            textTransform: "uppercase",
          }}
        >
          RAG · all-mpnet-base-v2 · chunk_384
        </span>
      </div>

      {/* Chat Area */}
      <div
        style={{
          paddingTop: "80px",
          paddingBottom: "140px",
          maxWidth: "760px",
          margin: "0 auto",
          padding: "80px 24px 140px",
          display: "flex",
          flexDirection: "column",
          gap: "24px",
          minHeight: "100vh",
        }}
      >
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} onShowChunks={handleShowChunks} />
        ))}
        {loading && (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: "6px", animation: "fadeSlideIn 0.3s ease forwards" }}>
            <div style={{ fontFamily: "'Space Mono', monospace", fontSize: "9px", color: "#484f58", letterSpacing: "1px", textTransform: "uppercase", paddingLeft: "4px" }}>
              DS Tutor
            </div>
            <div style={{ background: "#161b22", border: "1px solid #21262d", borderRadius: "16px 16px 16px 4px", padding: "12px 18px" }}>
              <TypingIndicator />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input Bar */}
      <div
        style={{
          position: "fixed",
          bottom: 0,
          left: 0,
          right: 0,
          padding: "16px 24px 24px",
          background: "linear-gradient(to top, #010409 60%, transparent)",
        }}
      >
        <div style={{ maxWidth: "760px", margin: "0 auto" }}>
        {books.length > 0 && (
          <BookSelector
            books={books}
            selected={selectedBooks}
            onChange={setSelectedBooks}
          />
        )}
        </div>
        <div
          style={{
            maxWidth: "760px",
            margin: "0 auto",
            display: "flex",
            gap: "10px",
            alignItems: "flex-end",
            background: "#161b22",
            border: "1px solid #30363d",
            borderRadius: "14px",
            padding: "12px 14px",
            transition: "border-color 0.2s",
          }}
          onFocusCapture={(e) => e.currentTarget.style.borderColor = "#64ffda55"}
          onBlurCapture={(e) => e.currentTarget.style.borderColor = "#30363d"}
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything from the book..."
            rows={1}
            style={{
              flex: 1,
              background: "none",
              border: "none",
              color: "#c9d1d9",
              fontFamily: "'IBM Plex Sans', sans-serif",
              fontSize: "14px",
              lineHeight: 1.6,
              maxHeight: "120px",
              overflowY: "auto",
            }}
            onInput={(e) => {
              e.target.style.height = "auto";
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
            }}
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || loading}
            style={{
              background: input.trim() && !loading ? "#64ffda" : "#21262d",
              border: "none",
              borderRadius: "8px",
              width: "34px",
              height: "34px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: input.trim() && !loading ? "pointer" : "not-allowed",
              transition: "all 0.2s",
              flexShrink: 0,
              color: input.trim() && !loading ? "#010409" : "#484f58",
              fontSize: "16px",
              fontWeight: "bold",
            }}
          >
            ↑
          </button>
        </div>
        <p style={{ textAlign: "center", fontFamily: "'Space Mono', monospace", fontSize: "9px", color: "#30363d", marginTop: "8px", letterSpacing: "1px" }}>
          ENTER to send · SHIFT+ENTER for newline · answers grounded in retrieved passages
        </p>
      </div>

      {/* Chunk Drawer */}
      <ChunkDrawer
        chunks={drawerChunks}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
      {drawerOpen && (
        <div
          onClick={() => setDrawerOpen(false)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.4)",
            zIndex: 99,
          }}
        />
      )}
    </>
  );
}
