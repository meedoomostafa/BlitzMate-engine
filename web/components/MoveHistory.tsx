"use client";

import React, { useEffect, useRef } from "react";

interface MoveHistoryProps {
  history: string[];
}

export default function MoveHistory({ history }: MoveHistoryProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history]);

  const pairs: Array<[number, string, string?]> = [];
  for (let i = 0; i < history.length; i += 2) {
    pairs.push([Math.floor(i / 2) + 1, history[i], history[i + 1]]);
  }

  return (
    <div className="history-panel">
      <h3 className="panel-title">Move History</h3>
      <div className="history-list" ref={scrollRef}>
        {pairs.length === 0 ? (
          <div className="history-empty">No moves played yet</div>
        ) : (
          <table className="history-table">
            <thead>
              <tr>
                <th>#</th>
                <th>White</th>
                <th>Black</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map(([num, w, b]) => (
                <tr key={num}>
                  <td>{num}.</td>
                  <td>{w}</td>
                  <td>{b || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
