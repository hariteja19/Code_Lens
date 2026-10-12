import "./globals.css";

export const metadata = {
  title: "CodeLens — RAG-Powered Code Intelligence",
  description:
    "Ask questions about any GitHub codebase. Get accurate answers with exact file and line citations. Powered by AST chunking, hybrid search with RRF, cross-encoder re-ranking, and Groq LLM inference.",
  authors: [{ name: "HariTeja" }],
  keywords: [
    "code search", "RAG", "code intelligence", "AI", "developer tools",
    "tree-sitter", "hybrid search", "qdrant", "vector database", "code understanding",
  ],
  openGraph: {
    title: "CodeLens — RAG-Powered Code Intelligence",
    description: "Ask questions about any GitHub codebase with exact file + line citations.",
    type: "website",
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
