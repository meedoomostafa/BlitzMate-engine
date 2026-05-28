"use client";

import { Chessboard } from "react-chessboard";
import EvalBar from "./EvalBar";

interface BoardPanelProps {
  fen: string;
  onPieceDrop: (sourceSquare: string, targetSquare: string) => boolean;
  thinking: boolean;
  score: number | null;
  gameOver: boolean;
  status: string;
  premove: { from: string; to: string; promotion?: string } | null;
  onSquareClick?: () => void;
  onSquareRightClick?: () => void;
}

export default function BoardPanel({
  fen,
  onPieceDrop,
  thinking,
  score,
  gameOver,
  status,
  premove,
  onSquareClick,
  onSquareRightClick,
}: BoardPanelProps) {
  // Highlight premove squares in soft red/orange
  const customSquareStyles = premove
    ? {
      [premove.from]: { backgroundColor: "rgba(224, 108, 117, 0.45)" },
      [premove.to]: { backgroundColor: "rgba(224, 108, 117, 0.45)" },
    }
    : {};

  return (
    <div className="board-wrapper">
      <div className="board-and-eval">
        <EvalBar score={score} gameOver={gameOver} status={status} />
        <div className="board-container">
          <Chessboard
            position={fen}
            onPieceDrop={onPieceDrop}
            arePiecesDraggable={!gameOver} // Enable dragging during opponent turn for premoves
            boardOrientation="white"
            animationDuration={100}
            customDarkSquareStyle={{ backgroundColor: "#8c7e73" }}
            customLightSquareStyle={{ backgroundColor: "#efeae4" }}
            customSquareStyles={customSquareStyles}
            onSquareClick={onSquareClick}
            onSquareRightClick={onSquareRightClick}
            customBoardStyle={{
              borderRadius: "6px",
              border: "1px solid var(--panel-border)",
              boxShadow: "0 12px 40px rgba(0, 0, 0, 0.4)",
            }}
          />
          {thinking && (
            <div className="board-thinking-indicator">
              <div className="spinner-mini"></div>
              <span>Thinking</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
