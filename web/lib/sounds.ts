"use client";

let isMutedGlobal = false;

export function setMutedGlobal(muted: boolean) {
  isMutedGlobal = muted;
}

function playSound(path: string) {
  if (isMutedGlobal) return;
  if (typeof window === "undefined") return;
  try {
    const audio = new Audio(path);
    audio.play().catch((e) => {
      if (e.name !== "AbortError") {
        console.warn("Audio play prevented:", e);
      }
    });
  } catch (e) {
    console.error("Failed to play sound:", e);
  }
}

export function playMoveSound() {
  playSound("/sounds/move.mp3");
}

export function playCaptureSound() {
  playSound("/sounds/capture.mp3");
}

export function playCheckSound() {
  playSound("/sounds/check.mp3");
}

export function playGameOverSound() {
  playSound("/sounds/game-over.mp3");
}

export function playIllegalSound() {
  playSound("/sounds/illegal.mp3");
}

export function playCastleSound() {
  playSound("/sounds/castle.mp3");
}

export function playPromoteSound() {
  playSound("/sounds/promote.mp3");
}

export function playSoundForMove(san: string, isGameOver: boolean) {
  if (isGameOver || san.includes("#")) {
    playGameOverSound();
  } else if (san.includes("+")) {
    playCheckSound();
  } else if (san.includes("O-O")) {
    playCastleSound();
  } else if (san.includes("=")) {
    playPromoteSound();
  } else if (san.includes("x")) {
    playCaptureSound();
  } else {
    playMoveSound();
  }
}
