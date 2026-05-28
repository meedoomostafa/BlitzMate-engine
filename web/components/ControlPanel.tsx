"use client";

import React from "react";

interface ControlPanelProps {
  depth: number;
  onDepthChange: (depth: number) => void;
  onReset: () => void;
  onSuggestMove: () => void;
  disabled: boolean;
  suggestion: { san: string; move: string } | null;
  muted: boolean;
  onToggleMute: () => void;
}

export default function ControlPanel({
  depth,
  onDepthChange,
  onReset,
  onSuggestMove,
  disabled,
  suggestion,
  muted,
  onToggleMute,
}: ControlPanelProps) {
  return (
    <div className="control-panel">
      <div className="panel-header-row">
        <h3 className="panel-title">Engine Options</h3>
        <button
          className={`mute-toggle-btn ${muted ? "muted" : ""}`}
          onClick={onToggleMute}
          aria-label={muted ? "Unmute game sounds" : "Mute game sounds"}
          title={muted ? "Unmute" : "Mute"}
        >
          {muted ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
              <line x1="23" y1="9" x2="17" y2="15" />
              <line x1="17" y1="9" x2="23" y2="15" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
            </svg>
          )}
        </button>
      </div>
      <div className="control-group">
        <label className="control-label">Search Depth (1-6)</label>
        <div className="depth-selector">
          {[1, 2, 3, 4, 5, 6].map((d) => (
            <button
              key={d}
              className={`depth-btn ${depth === d ? "active" : ""}`}
              onClick={() => onDepthChange(d)}
              disabled={disabled}
            >
              {d}
            </button>
          ))}
        </div>
      </div>
      <div className="action-buttons">
        <button
          className="btn btn-secondary"
          onClick={onSuggestMove}
          disabled={disabled}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: "6px", display: "inline-block", verticalAlign: "text-bottom" }}>
            <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A5 5 0 0 0 8 8c0 1.3.5 2.6 1.5 3.5.8.8 1.3 1.5 1.5 2.5" />
            <path d="M9 18h6M10 22h4" />
          </svg>
          Suggest Move
        </button>
        <button
          className="btn btn-primary"
          onClick={onReset}
          disabled={disabled}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: "6px", display: "inline-block", verticalAlign: "text-bottom" }}>
            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
            <path d="M16 3h5v5M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
            <path d="M8 21H3v-5" />
          </svg>
          New Game
        </button>
      </div>
    </div>
  );
}
