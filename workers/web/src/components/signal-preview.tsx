"use client";

import Image from "next/image";
import { useState } from "react";

const signals = [
  { name: "VIDEO", color: "#77a5ff", phase: 0.4 },
  { name: "AUDIO", color: "#97d5b3", phase: 2.1 },
  { name: "TEXT", color: "#ff8a76", phase: 4.2 },
];

function waveform(phase: number) {
  return Array.from({ length: 121 }, (_, i) => {
    const envelope = 0.3 + 0.7 * Math.sin(i / 21 + phase) ** 2;
    const y =
      20 +
      (Math.sin(i * 0.34 + phase) * 10 + Math.sin(i * 0.81) * 5) * envelope;
    return `${i === 0 ? "M" : "L"}${i * 4},${y.toFixed(2)}`;
  }).join(" ");
}

export function SignalPreview() {
  const [mode, setMode] = useState<"activity" | "multimodal">("activity");

  return (
    <figure
      className="signal-preview"
      aria-label="Illustration of Previral’s brain activity views"
    >
      <div className="preview-topline mono">
        <span>
          <i className="status-dot" /> CORTICAL VIEW
        </span>
        <span>TRIBE v2</span>
      </div>
      <div
        className="view-switch"
        role="group"
        aria-label="Brain visualization mode"
      >
        <button
          type="button"
          aria-pressed={mode === "activity"}
          onClick={() => setMode("activity")}
        >
          Activity
        </button>
        <button
          type="button"
          aria-pressed={mode === "multimodal"}
          onClick={() => setMode("multimodal")}
        >
          Multimodality
        </button>
      </div>
      <div className="brain-stage">
        <div className="crosshair crosshair-left" aria-hidden="true" />
        <div className="crosshair crosshair-right" aria-hidden="true" />
        <Image
          className={
            mode === "activity" ? "brain-image visible" : "brain-image"
          }
          src="/brain-activity.png"
          alt="Cortical surface with illustrative orange and yellow activation regions"
          width={1000}
          height={1000}
          priority
          aria-hidden={mode !== "activity"}
        />
        <Image
          className={
            mode === "multimodal" ? "brain-image visible" : "brain-image"
          }
          src="/brain-multimodal.png"
          alt="Cortical surface with illustrative video, audio, and text contributions in blue, green, and red"
          width={1000}
          height={1000}
          aria-hidden={mode !== "multimodal"}
        />
        <div className="hemisphere hemisphere-left mono" aria-hidden="true">
          L
        </div>
        <div className="hemisphere hemisphere-right mono" aria-hidden="true">
          R
        </div>
        <div className="brain-caption mono" aria-live="polite">
          {mode === "activity" ? "PREDICTED ACTIVITY" : "VIDEO + AUDIO + TEXT"}
          <span>20,484 VERTICES</span>
        </div>
      </div>
      <div className="signal-tracks" aria-hidden="true">
        <div className="track-ruler mono">
          <span>00:00</span>
          <span>00:15</span>
          <span>00:30</span>
          <span>00:45</span>
        </div>
        {signals.map((signal) => (
          <div className="signal-track" key={signal.name}>
            <span className="mono" style={{ color: signal.color }}>
              {signal.name}
            </span>
            <svg viewBox="0 0 480 40" preserveAspectRatio="none">
              <path
                d={waveform(signal.phase)}
                stroke={signal.color}
                fill="none"
                strokeWidth="1.3"
              />
            </svg>
          </div>
        ))}
        <div className="playhead" />
      </div>
      <figcaption className="preview-footnote mono">
        <span className="tiny-cross">+</span> INTERACTIVE ILLUSTRATION ·
        SYNTHETIC SIGNALS
      </figcaption>
    </figure>
  );
}
