"use client";

import React from "react";

interface StatusBarProps {
  status: string;
  game_over: boolean;
  thinking: boolean;
  score: number | null;
  depth: number;
  suggestion: { san: string; move: string } | null;
}

export default function StatusBar({
  status,
  game_over,
  thinking,
  score,
  depth,
  suggestion,
}: StatusBarProps) {
  const getStatusText = () => {
    if (thinking) return "BlitzMate is searching...";
    if (game_over) {
      if (status === "checkmate") return "Checkmate! Game over.";
      if (status === "stalemate") return "Stalemate! Game drawn.";
      if (status === "draw") return "Draw! Game over.";
      return "Game Over";
    }
    if (status === "check") return "Check! Your move.";
    if (status === "white_to_move") return "Your turn (White)";
    if (status === "black_to_move") return "BlitzMate's turn (Black)";
    return "Ready";
  };

  const formatScore = () => {
    if (score === null) return "0.00";
    const val = score / 100;
    const sign = val > 0 ? "+" : "";
    return `${sign}${val.toFixed(2)}`;
  };

  return (
    <div className="status-bar">
      <div className="status-item">
        <span className="status-label">Status</span>
        <span className={`status-value ${thinking ? "text-thinking" : ""}`}>
          {getStatusText()}
        </span>
      </div>
      <div className="status-item">
        <span className="status-label">Eval</span>
        <span
          className={`status-value ${
            score !== null && score > 0
              ? "text-good"
              : score !== null && score < 0
              ? "text-bad"
              : ""
          }`}
        >
          {formatScore()}
        </span>
      </div>
      <div className="status-item">
        <span className="status-label">Depth</span>
        <span className="status-value">{depth}</span>
      </div>
      <div className="status-item">
        <span className="status-label">Suggested Move</span>
        <span className={`status-value ${suggestion ? "suggestion-text" : ""}`}>
          {suggestion ? `${suggestion.san} (${suggestion.move})` : "—"}
        </span>
      </div>
    </div>
  );
}
