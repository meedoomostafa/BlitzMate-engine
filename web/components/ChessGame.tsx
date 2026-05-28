"use client";

import React, { useState, useRef, useEffect } from "react";
import { Chess, Square } from "chess.js";
import { playMove, getBestMove } from "../lib/api";
import { playSoundForMove, playIllegalSound, setMutedGlobal } from "../lib/sounds";
import BoardPanel from "./BoardPanel";
import StatusBar from "./StatusBar";
import MoveHistory from "./MoveHistory";
import ControlPanel from "./ControlPanel";
import GameResultModal from "./GameResultModal";

export default function ChessGame() {
  const [game, setGame] = useState(() => new Chess());
  const [history, setHistory] = useState<string[]>([]);
  const [searchDepth, setSearchDepth] = useState(5); // Default 5
  const [thinking, setThinking] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const [gameStatus, setGameStatus] = useState("white_to_move");
  const [score, setScore] = useState<number | null>(0);
  const [error, setError] = useState<string | null>(null);
  const [suggestion, setSuggestion] = useState<{ san: string; move: string } | null>(null);

  // Premove states
  const [premove, setPremoveState] = useState<{ from: string; to: string; promotion?: string } | null>(null);
  const [premoveFen, setPremoveFen] = useState<string | null>(null);
  const premoveRef = useRef(premove);

  // Click-to-move states
  const [selectedSquare, setSelectedSquare] = useState<string | null>(null);
  const [legalMoves, setLegalMoves] = useState<string[]>([]);

  // Audio and Victory Modal states
  const [muted, setMuted] = useState(false);
  const [showResultModal, setShowResultModal] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("blitzmate_muted");
      if (saved !== null) {
        setTimeout(() => {
          setMuted(saved === "true");
        }, 0);
      }
    }
  }, []);

  useEffect(() => {
    setMutedGlobal(muted);
  }, [muted]);

  const toggleMute = () => {
    const next = !muted;
    setMuted(next);
    localStorage.setItem("blitzmate_muted", String(next));
  };

  const getWinner = (): "user" | "engine" | "draw" => {
    if (!gameOver) return "draw";
    if (gameStatus === "checkmate") {
      return game.turn() === "b" ? "user" : "engine";
    }
    return "draw";
  };

  const setPremove = (val: typeof premove) => {
    premoveRef.current = val;
    setPremoveState(val);
  };

  const cancelPremove = () => {
    if (premoveRef.current && premoveFen) {
      setGame(new Chess(premoveFen));
      setPremove(null);
      setPremoveFen(null);
    }
  };

  const clearSelection = () => {
    setSelectedSquare(null);
    setLegalMoves([]);
  };

  const handleSquareClick = (square: string, piece: string | undefined) => {
    if (gameOver) return;

    // If a square is already selected and user clicks a legal target → execute move
    if (selectedSquare && legalMoves.includes(square)) {
      onPieceDrop(selectedSquare, square);
      clearSelection();
      return;
    }

    // If clicked square has a White piece → select it
    if (piece && piece.startsWith("w")) {
      const baseFen = premoveFen || game.fen();
      const tokens = baseFen.split(" ");
      tokens[1] = "w";
      const gameCopy = new Chess(tokens.join(" "));
      try {
        const moves = gameCopy.moves({ square: square as Square, verbose: true });
        const targets = moves.map((m) => m.to);
        setSelectedSquare(square);
        setLegalMoves(targets);
      } catch {
        clearSelection();
      }
    } else {
      clearSelection();
    }
  };

  const resetGame = () => {
    setGame(new Chess());
    setHistory([]);
    setGameOver(false);
    setGameStatus("white_to_move");
    setScore(0);
    setError(null);
    setSuggestion(null);
    setPremove(null);
    setPremoveFen(null);
    setShowResultModal(false);
    clearSelection();
  };

  const handleSuggestMove = async () => {
    if (thinking || gameOver) return;
    setThinking(true);
    setError(null);
    try {
      const res = await getBestMove({
        fen: game.fen(),
        depth: searchDepth,
      });
      if (res.move && res.san) {
        setSuggestion({ san: res.san, move: res.move });
      } else {
        setError("No moves suggested.");
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      setError(errMsg);
    } finally {
      setThinking(false);
    }
  };

  const triggerEngineMove = (previousFen: string, userMoveStr: string) => {
    setThinking(true);
    playMove({
      fen: previousFen,
      user_move: userMoveStr,
      depth: searchDepth,
    })
      .then((response) => {
        if (!response.legal) {
          setGame(new Chess(previousFen));
          setHistory((prev) => prev.slice(0, -1));
          setError(response.error || "Illegal move");
          setPremove(null);
          setPremoveFen(null);
          setThinking(false);
          playIllegalSound();
          return;
        }

        const nextGame = new Chess(response.fen);
        setGame(nextGame);

        if (response.engine_san) {
          setHistory((prev) => [...prev, response.engine_san!]);
          playSoundForMove(response.engine_san, response.game_over);
        } else if (response.game_over) {
          playSoundForMove("", true);
        }

        setGameOver(response.game_over);
        setGameStatus(response.status);
        if (response.game_over) {
          setShowResultModal(true);
        }

        // Fetch score evaluation
        fetchEval(response.fen);

        // Check if there is a pending premove
        const currentPremove = premoveRef.current;
        if (currentPremove && !response.game_over) {
          const freshGame = new Chess(response.fen);
          try {
            const premoveObj = freshGame.move({
              from: currentPremove.from,
              to: currentPremove.to,
              promotion: currentPremove.promotion,
            });

            if (premoveObj) {
              const currentFen = response.fen;
              setGame(freshGame);
              setHistory((prev) => [...prev, premoveObj.san]);
              playSoundForMove(premoveObj.san, false);
              setPremove(null);
              setPremoveFen(null);
              // Trigger next engine move
              triggerEngineMove(currentFen, premoveObj.from + premoveObj.to + (premoveObj.promotion || ""));
              return;
            }
          } catch {
            // Premove is illegal in the new position, just discard it
          }
        }

        setPremove(null);
        setPremoveFen(null);
        setThinking(false);
      })
      .catch((err) => {
        setGame(new Chess(previousFen));
        setHistory((prev) => prev.slice(0, -1));
        setError(err.message || "Engine API communication error");
        setPremove(null);
        setPremoveFen(null);
        setThinking(false);
      });
  };

  const onPieceDrop = (sourceSquare: string, targetSquare: string) => {
    if (gameOver) return false;

    if (thinking) {
      const baseFen = premoveFen || game.fen();
      const tokens = baseFen.split(" ");
      tokens[1] = "w";
      const gameCopy = new Chess(tokens.join(" "));

      // Check if source has a White piece (permissive premove validation)
      const sourcePiece = gameCopy.get(sourceSquare as Square);
      if (!sourcePiece || sourcePiece.color !== "w" || sourceSquare === targetSquare) {
        playIllegalSound();
        clearSelection();
        return false;
      }

      // Try strict validation first
      try {
        const isPromotion =
          sourcePiece.type === "p" &&
          ((gameCopy.turn() === "w" && targetSquare[1] === "8") ||
            (gameCopy.turn() === "b" && targetSquare[1] === "1"));

        const moveObj = gameCopy.move({
          from: sourceSquare,
          to: targetSquare,
          promotion: isPromotion ? "q" : undefined,
        });

        if (moveObj) {
          setPremove({
            from: sourceSquare,
            to: targetSquare,
            promotion: isPromotion ? "q" : undefined,
          });
          setPremoveFen(baseFen);
          setGame(gameCopy);
          clearSelection();
          return true;
        }
      } catch {
        // Strict validation failed, but permissive premove allows it
      }

      // Permissive premove: accept the move even if strict validation fails
      // (e.g., captures that are illegal now but may be legal after engine moves)
      const isPromotion =
        sourcePiece.type === "p" &&
        ((gameCopy.turn() === "w" && targetSquare[1] === "8") ||
          (gameCopy.turn() === "b" && targetSquare[1] === "1"));

      setPremove({
        from: sourceSquare,
        to: targetSquare,
        promotion: isPromotion ? "q" : undefined,
      });
      setPremoveFen(baseFen);
      clearSelection();
      return true;
    }


    const gameCopy = new Chess(game.fen());
    let moveObj = null;
    try {
      const isPromotion =
        gameCopy.get(sourceSquare as Square)?.type === "p" &&
        ((gameCopy.turn() === "w" && targetSquare[1] === "8") ||
          (gameCopy.turn() === "b" && targetSquare[1] === "1"));

      moveObj = gameCopy.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: isPromotion ? "q" : undefined,
      });
    } catch {
      playIllegalSound();
      clearSelection();
      return false;
    }

    if (!moveObj) {
      playIllegalSound();
      clearSelection();
      return false;
    }

    const previousFen = game.fen();
    setGame(gameCopy);
    setHistory((prev) => [...prev, moveObj.san]);
    setError(null);
    setSuggestion(null);
    clearSelection();

    playSoundForMove(moveObj.san, false);

    triggerEngineMove(previousFen, moveObj.from + moveObj.to + (moveObj.promotion || ""));

    return true;
  };

  const fetchEval = async (fen: string) => {
    try {
      const res = await getBestMove({ fen, depth: 1 });
      setScore(res.score_cp);
    } catch {
      // Keep existing evaluation
    }
  };

  return (
    <div className="game-layout">
      <div className="left-column">
        <BoardPanel
          fen={game.fen()}
          onPieceDrop={onPieceDrop}
          thinking={thinking}
          score={score}
          gameOver={gameOver}
          status={gameStatus}
          premove={premove}
          selectedSquare={selectedSquare}
          legalMoves={legalMoves}
          onSquareClick={handleSquareClick}
          onSquareRightClick={() => {
            cancelPremove();
            clearSelection();
          }}
        />
        <StatusBar
          status={gameStatus}
          game_over={gameOver}
          thinking={thinking}
          score={score}
          depth={searchDepth}
          suggestion={suggestion}
        />
        {error && (
          <div className="error-banner">
            <span className="banner-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </span>
            <span>{error}</span>
          </div>
        )}
      </div>
      <div className="right-column">
        <ControlPanel
          depth={searchDepth}
          onDepthChange={setSearchDepth}
          onReset={resetGame}
          onSuggestMove={handleSuggestMove}
          disabled={thinking}
          suggestion={suggestion}
          muted={muted}
          onToggleMute={toggleMute}
        />
        <MoveHistory history={history} />
      </div>

      <GameResultModal
        isOpen={showResultModal}
        onClose={() => setShowResultModal(false)}
        onPlayAgain={resetGame}
        winner={getWinner()}
        status={gameStatus}
        movesCount={Math.ceil(history.length / 2)}
      />
    </div>
  );
}
