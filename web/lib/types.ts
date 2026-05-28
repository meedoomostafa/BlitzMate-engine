export interface BestMoveRequest {
  fen: string;
  depth?: number;
}

export interface BestMoveResponse {
  move: string | null;
  san: string | null;
  depth: number;
  score_cp: number;
  nodes: number;
  time_ms: number;
  game_over: boolean;
  status: string;
}

export interface PlayRequest {
  fen: string;
  user_move: string;
  depth?: number;
}

export interface PlayResponse {
  fen: string;
  engine_move: string | null;
  engine_san: string | null;
  legal: boolean;
  game_over: boolean;
  status: string;
  error: string | null;
}

export interface HealthResponse {
  status: string;
  engine: string;
  max_depth: number;
}
