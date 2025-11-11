export function AudioCard({
  coughCount,
  audioStatus,
  audioResult,
  isMonitoring,
  onStart,
  onStop,
}) {
  return (
    <div className="bg-slate-800 p-6 rounded-2xl shadow-lg text-center">
      <h2 className="text-2xl font-semibold text-cyan-400 mb-4">Audio Event Monitor</h2>

      <p className="text-8xl font-bold mb-2">{coughCount}</p>
      <p className="text-sm text-slate-400 mb-4">Cough Count</p>

      {audioResult && (
        <p className="text-lg font-medium mb-2">
          {audioResult.prediction} ({(audioResult.confidence * 100).toFixed(1)}%)
        </p>
      )}

      <p className="text-slate-400 text-sm mb-4">{audioStatus}</p>

      <button
        onClick={isMonitoring ? onStop : onStart}
        className={`px-8 py-3 rounded-full font-semibold transition ${
          isMonitoring
            ? "bg-red-500 hover:bg-red-600"
            : "bg-green-500 hover:bg-green-600"
        }`}
      >
        {isMonitoring ? "Stop Monitoring" : "Start Monitoring"}
      </button>
    </div>
  );
}
