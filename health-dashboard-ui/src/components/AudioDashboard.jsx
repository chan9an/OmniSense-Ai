import React from "react";
import { useAudioMonitor } from "../hooks/useAudioMonitor";

export default function AudioDashboard({ apiUrl }) {
  const { isMonitoring, startMonitoring, stopMonitoring, prediction, coughCount } =
    useAudioMonitor(apiUrl);

  return (
    <div className="p-6 bg-slate-800 text-white rounded-2xl shadow-xl">
      <h2 className="text-2xl font-bold mb-4">Audio Event Monitor</h2>
      <p className="text-lg mb-2">Cough Count: {coughCount}</p>
      {prediction && (
        <p className="mb-4 text-cyan-300">
          {prediction.prediction} ({Math.round(prediction.confidence * 100)}%)
        </p>
      )}
      <button
        onClick={isMonitoring ? stopMonitoring : startMonitoring}
        className={`px-6 py-2 rounded-full font-semibold transition ${
          isMonitoring ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"
        }`}
      >
        {isMonitoring ? "Stop Monitoring" : "Start Monitoring"}
      </button>
    </div>
  );
}
