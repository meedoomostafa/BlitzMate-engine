"use client";

import { useEffect } from "react";

interface GameResultModalProps {
  isOpen: boolean;
  onClose: () => void;
  onPlayAgain: () => void;
  winner: "user" | "engine" | "draw";
  status: string;
  movesCount: number;
}

export default function GameResultModal({
  isOpen,
  onClose,
  onPlayAgain,
  winner,
  status,
  movesCount,
}: GameResultModalProps) {
  useEffect(() => {
    if (isOpen && winner === "user") {
      import("canvas-confetti").then((module) => {
        const confetti = module.default;
        const colors = ["#c9b9a9", "#efeae4", "#8b9986", "#e5c158"];
        
        // Initial bursts from sides
        confetti({
          particleCount: 60,
          angle: 60,
          spread: 60,
          origin: { x: 0, y: 0.8 },
          colors: colors,
        });
        confetti({
          particleCount: 60,
          angle: 120,
          spread: 60,
          origin: { x: 1, y: 0.8 },
          colors: colors,
        });

        // Soft shower from top center
        setTimeout(() => {
          confetti({
            particleCount: 40,
            angle: 90,
            spread: 80,
            origin: { x: 0.5, y: 0.3 },
            colors: colors,
          });
        }, 300);
      });
    }
  }, [isOpen, winner]);

  if (!isOpen) return null;

  const getTitle = () => {
    if (winner === "user") return "Brilliant Victory";
    if (winner === "engine") return "Defeat";
    return "Draw Declared";
  };

  const getSubtitle = () => {
    if (winner === "user") return "You outplayed the engine with masterclass precision.";
    if (winner === "engine") return "A calculated checkmate by the BlitzMate engine.";
    if (status === "stalemate") return "No legal moves remaining. Stalemate.";
    return "The game ended in a draw.";
  };

  const getResultCode = () => {
    if (winner === "user") return "1 - 0";
    if (winner === "engine") return "0 - 1";
    return "½ - ½";
  };

  return (
    <div className="modal-backdrop">
      <div className={`modal-card winner-${winner}`}>
        {/* Status Icon */}
        <div className="modal-icon-wrapper">
          {winner === "user" && (
            <div className="trophy-glow-effect">
              <svg className="modal-icon gold-glow" width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.25">
                <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6" />
                <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18" />
                <path d="M4 22h16" />
                <path d="M10 14.66V17c0 .55-.45 1-1 1H4v2h16v-2h-5c-.55 0-1-.45-1-1v-2.34" />
                <path d="M12 2a7 7 0 0 0-7 7c0 2.25.75 4 2 5.34l.32.32a4.98 4.98 0 0 0 9.36 0l.32-.32a6.98 6.98 0 0 0 2-5.34 7 7 0 0 0-7-7Z" />
              </svg>
            </div>
          )}
          {winner === "engine" && (
            <svg className="modal-icon red-glow" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 2a3 3 0 0 0-3 3v2a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
              <path d="M19 13V10a7 7 0 0 0-14 0v3" />
              <path d="M5 13a3 3 0 0 0-3 3v2a3 3 0 0 0 6 0v-2a3 3 0 0 0-3-3Z" />
              <path d="M19 13a3 3 0 0 0-3 3v2a3 3 0 0 0 6 0v-2a3 3 0 0 0-3-3Z" />
              <path d="M12 10v4" />
              <path d="M8 21h8" />
            </svg>
          )}
          {winner === "draw" && (
            <svg className="modal-icon bronze-glow" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M16 3H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2Z" />
              <path d="M12 3v18" />
              <path d="M6 12h12" />
            </svg>
          )}
        </div>

        <h2 className="modal-title">{getTitle()}</h2>
        <p className="modal-subtitle">{getSubtitle()}</p>

        {/* Stats Grid */}
        <div className="modal-stats">
          <div className="modal-stat-item">
            <span className="modal-stat-label">Result</span>
            <span className="modal-stat-value">{getResultCode()}</span>
          </div>
          <div className="modal-stat-item">
            <span className="modal-stat-label">Moves Played</span>
            <span className="modal-stat-value">{movesCount}</span>
          </div>
          <div className="modal-stat-item">
            <span className="modal-stat-label">Ending Condition</span>
            <span className="modal-stat-value text-capitalize">{status.replace("_", " ")}</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="modal-actions">
          <button className="btn btn-primary btn-lg" onClick={onPlayAgain}>
            Play Again
          </button>
          <button className="btn btn-secondary btn-lg" onClick={onClose}>
            Review Board
          </button>
        </div>
      </div>
    </div>
  );
}
