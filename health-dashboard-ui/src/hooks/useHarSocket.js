import { useState, useEffect, useRef } from "react";
import io from "socket.io-client";

const HAR_LABELS = ["Downstairs", "Jogging", "Still", "Upstairs", "Walking"];

export function useHarSocket(NGROK_URL) {
  const [harConnected, setHarConnected] = useState(false);
  const [harActivity, setHarActivity] = useState("Waiting...");
  const [harConfidence, setHarConfidence] = useState(0);
  const [activityTimes, setActivityTimes] = useState(() =>
    HAR_LABELS.reduce((acc, l) => ({ ...acc, [l]: 0 }), {})
  );

  const currentActivityRef = useRef("Waiting...");
  const lastActivityStart = useRef(Date.now());

  useEffect(() => {
    const socket = io.connect(NGROK_URL, { transports: ["websocket"] });

    socket.on("connect", () => setHarConnected(true));
    socket.on("disconnect", () => setHarConnected(false));

    socket.on("har_prediction", (data) => {
      if (data.activity !== currentActivityRef.current) {
        const duration = Date.now() - lastActivityStart.current;
        setActivityTimes((prev) => ({
          ...prev,
          [currentActivityRef.current]:
            prev[currentActivityRef.current] + duration,
        }));
        currentActivityRef.current = data.activity;
        lastActivityStart.current = Date.now();
      }
      setHarActivity(data.activity);
      setHarConfidence(data.confidence);
    });

    return () => socket.disconnect();
  }, [NGROK_URL]);

  return { harConnected, harActivity, harConfidence, activityTimes };
}
