import { useState, useRef, useEffect } from "react";

export function useAudioMonitor(AUDIO_API_URL) {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [audioStatus, setAudioStatus] = useState("Idle");
  const [audioResult, setAudioResult] = useState(null);
  const [coughCount, setCoughCount] = useState(0);
  const [showModal, setShowModal] = useState(false);

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);

  const COUGH_CONF_THRESHOLD = 0.4;
  const RECORD_MS = 5200;

  useEffect(() => {
    return () => stopMonitoring();
  }, []);

  const startMonitoring = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setIsMonitoring(true);
      setAudioStatus("🎙 Starting...");
      loopRecord();
    } catch (err) {
      console.error("❌ Mic denied:", err);
      setAudioStatus("Mic blocked");
    }
  };

  const stopMonitoring = () => {
    setIsMonitoring(false);
    if (mediaRecorderRef.current?.state === "recording")
      mediaRecorderRef.current.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setAudioStatus("Idle");
  };

  const loopRecord = () => {
    if (!isMonitoring || !streamRef.current) return;
    const rec = new MediaRecorder(streamRef.current);
    mediaRecorderRef.current = rec;
    let chunks = [];
    setAudioStatus("🔴 Recording...");

    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = async () => {
      setAudioStatus("🧠 Analyzing...");
      const blob = new Blob(chunks, { type: "audio/wav" });
      const file = new File([blob], "recording.wav", { type: "audio/wav" });
      const result = await analyze(file);
      console.log("🎧 Model Output:", result);

      if (result) {
        const { prediction, confidence } = result;
        if (prediction === "coughing" && confidence >= COUGH_CONF_THRESHOLD) {
          setCoughCount((prev) => prev + 1);
          setAudioResult({ label: "Cough Detected", confidence });
          setAudioStatus(`🤧 Cough (${Math.round(confidence * 100)}%)`);
          if (coughCount >= 5) setShowModal(true);
        } else {
          setAudioResult({ label: prediction, confidence });
          setAudioStatus(`${prediction} (${Math.round(confidence * 100)}%)`);
        }
      } else {
        setAudioStatus("❌ No prediction");
      }

      if (isMonitoring) setTimeout(loopRecord, 100);
    };

    rec.start();
    setTimeout(() => rec.state === "recording" && rec.stop(), RECORD_MS);
  };

  const analyze = async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(AUDIO_API_URL, { method: "POST", body: fd });
      const data = await res.json();
      return res.ok ? data : null;
    } catch (e) {
      console.error("❌ Backend unreachable:", e);
      stopMonitoring();
      return null;
    }
  };

  return {
    isMonitoring,
    audioStatus,
    audioResult,
    coughCount,
    showModal,
    setShowModal,
    startMonitoring,
    stopMonitoring,
  };
}
