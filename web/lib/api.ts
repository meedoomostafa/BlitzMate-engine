import { BestMoveRequest, BestMoveResponse, PlayRequest, PlayResponse, HealthResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_ENGINE_API_URL || "http://127.0.0.1:7860";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });

  if (!response.ok) {
    let errorMessage = `HTTP error! Status: ${response.status}`;
    try {
      const errorData = await response.json();
      if (errorData && errorData.detail) {
        if (typeof errorData.detail === "string") {
          errorMessage = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail
            .map((err: { loc: (string | number)[]; msg: string }) => `${err.loc.join(".")}: ${err.msg}`)
            .join("; ");
        }
      }
    } catch {
      // Ignore if response is not JSON
    }
    throw new Error(errorMessage);
  }

  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export async function getBestMove(req: BestMoveRequest): Promise<BestMoveResponse> {
  return request<BestMoveResponse>("/api/best-move", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function playMove(req: PlayRequest): Promise<PlayResponse> {
  return request<PlayResponse>("/api/play", {
    method: "POST",
    body: JSON.stringify(req),
  });
}
