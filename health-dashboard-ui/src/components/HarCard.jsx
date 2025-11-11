export function HarCard({ harConnected, harActivity, harConfidence, activityTimes }) {
  return (
    <div className="bg-slate-800 p-6 rounded-2xl shadow-lg text-center">
      <h2 className="text-2xl font-semibold text-indigo-400 mb-4">Activity Recognition</h2>
      <p className="text-4xl font-bold">{harActivity}</p>
      <p className="text-sm text-slate-400">
        Confidence: {(harConfidence * 100).toFixed(1)}%
      </p>
      <div className="mt-6">
        <h3 className="text-indigo-400 font-semibold mb-2">Session Tracker</h3>
        <ul className="text-left text-slate-300 text-sm space-y-1">
          {Object.entries(activityTimes).map(([a, t]) => (
            <li key={a}>
              {a}: {(t / 1000).toFixed(0)}s
            </li>
          ))}
        </ul>
      </div>
      <p className={`mt-4 ${harConnected ? "text-green-400" : "text-red-400"}`}>
        {harConnected ? "Connected" : "Disconnected"}
      </p>
    </div>
  );
}
