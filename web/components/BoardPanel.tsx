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
  selectedSquare: string | null;
  legalMoves: string[];
  onSquareClick?: (square: string, piece: string | undefined) => void;
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
  selectedSquare,
  legalMoves,
  onSquareClick,
  onSquareRightClick,
}: BoardPanelProps) {
  const position = Object.fromEntries(
    fen
      .split(" ")[0]
      .split("/")
      .flatMap((rank, rankIdx) =>
        [...rank].reduce(
          (acc: { squares: [string, string | null][]; fileIdx: number }, ch) => {
            if (/\d/.test(ch)) {
              for (let i = 0; i < parseInt(ch); i++) {
                acc.squares.push([
                  `${String.fromCharCode(97 + acc.fileIdx + i)}${8 - rankIdx}`,
                  null,
                ]);
              }
              acc.fileIdx += parseInt(ch);
            } else {
              acc.squares.push([
                `${String.fromCharCode(97 + acc.fileIdx)}${8 - rankIdx}`,
                ch,
              ]);
              acc.fileIdx += 1;
            }
            return acc;
          },
          { squares: [], fileIdx: 0 }
        ).squares
      )
  ) as Record<string, string | null>;

  const customSquareStyles: Record<string, Record<string, string>> = {};

  if (premove) {
    customSquareStyles[premove.from] = { backgroundColor: "rgba(224, 108, 117, 0.45)" };
    customSquareStyles[premove.to] = { backgroundColor: "rgba(224, 108, 117, 0.45)" };
  }

  if (selectedSquare) {
    customSquareStyles[selectedSquare] = { backgroundColor: "rgba(100, 200, 100, 0.5)" };
  }

  legalMoves.forEach((sq) => {
    const piece = position[sq];
    if (piece) {
      customSquareStyles[sq] = {
        background:
          "radial-gradient(circle, transparent 55%, rgba(220, 50, 50, 0.5) 55%)",
      };
    } else {
      customSquareStyles[sq] = {
        background:
          "radial-gradient(circle, rgba(100, 200, 100, 0.5) 25%, transparent 25%)",
      };
    }
  });

  return (
    <div className="board-wrapper">
      <div className="board-and-eval">
        <EvalBar score={score} gameOver={gameOver} status={status} />
        <div className="board-container">
          <Chessboard
            position={fen}
            onPieceDrop={onPieceDrop}
            arePiecesDraggable={!gameOver}
            boardOrientation="white"
            animationDuration={100}
            customDarkSquareStyle={{ backgroundColor: "#8c7e73" }}
            customLightSquareStyle={{ backgroundColor: "#efeae4" }}
            customSquareStyles={customSquareStyles}
            onSquareClick={(square, piece) => onSquareClick?.(square, piece)}
            onSquareRightClick={() => onSquareRightClick?.()}
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
