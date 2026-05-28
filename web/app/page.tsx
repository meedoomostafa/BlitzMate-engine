import ChessGame from "../components/ChessGame";

export default function Home() {
  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-left">
          <div className="header-logo">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="header-logo-svg">
              <path d="M4 6h16l-1.5 8h-13L4 6z" />
              <path d="M9 14v4h6v-4" />
              <path d="M7 6l1.5-3 1.5 3 2-3 2 3 1.5-3 1.5 3" />
              <path d="M3 20h18" />
            </svg>
          </div>
          <div className="header-text">
            <h1>BlitzMate</h1>
            <p>Play Chess Against BlitzMate Engine</p>
          </div>
        </div>
        <a 
          href="https://github.com/meedoomostafa/BlitzMate-engine" 
          target="_blank" 
          rel="noopener noreferrer"
          className="github-link"
          aria-label="GitHub Repository"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="github-icon">
            <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />
          </svg>
        </a>
      </header>

      <main className="app-main">
        <ChessGame />
      </main>

      <footer className="app-footer">
        <p>
          Powered by BlitzMate Engine &copy; 2026.
        </p>
      </footer>
    </div>
  );
}
