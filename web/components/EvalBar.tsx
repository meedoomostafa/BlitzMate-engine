"use client";

import React from "react";

interface EvalBarProps {
  score: number | null;
  gameOver: boolean;
  status: string;
}

export default function EvalBar({ score, gameOver, status }: EvalBarProps) {
  const val = score === null ? 0 : score;
  let whitePercentage = 50;

  if (gameOver) {
    if (status === "checkmate") {
      whitePercentage = val >= 0 ? 100 : 0;
    } else {
      whitePercentage = 50;
    }
  } else {
    const pawns = val / 100;
    whitePercentage = 50 + (pawns * 5); // +5 pawns -> 75%, -5 pawns -> 25%
    whitePercentage = Math.max(5, Math.min(95, whitePercentage));
  }

  const formatScoreLabel = () => {
    if (score === null) return "0.0";
    const pawns = Math.abs(score / 100);
    return pawns.toFixed(1);
  };

  const isWhiteWinning = (score || 0) >= 0;

  return (
    <div className="eval-bar-container">
      <div 
        className="eval-bar-white" 
        style={{ height: `${whitePercentage}%` }}
      >
        {isWhiteWinning && <span className="eval-score-text text-black-label">{formatScoreLabel()}</span>}
      </div>
      <div className="eval-bar-black">
        {!isWhiteWinning && <span className="eval-score-text text-white-label">{formatScoreLabel()}</span>}
      </div>
    </div>
  );
}
