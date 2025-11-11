// src/App.jsx
import { motion } from "framer-motion";
import { useHarSocket } from "./hooks/useHarSocket";
import { useAudioMonitor } from "./hooks/useAudioMonitor";
import { HarCard } from "./components/HarCard";
import { AudioCard } from "./components/AudioCard";
import { Modal } from "./components/Modal";

// ======================================================
// === CONFIGURATION ====================================
// ======================================================
const NGROK_URL = "https://stormily-monaxial-tabetha.ngrok-free.dev"; // your Flask server ngrok URL
const AUDIO_API_URL = `${NGROK_URL}/predict_audio`;

// ======================================================
// === MAIN APP =========================================
// ======================================================
function App() {
  const { harConnected, harActivity, harConfidence, activityTimes } = useHarSocket(NGROK_URL);

  const {
    isMonitoring,
    coughCount,
    audioStatus,
    audioResult,
    showModal,
    setShowModal,
    startMonitoring,
    stopMonitoring,
  } = useAudioMonitor(AUDIO_API_URL);

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-slate-900 via-indigo-950 to-black text-white flex flex-col items-center justify-center p-6 md:p-10">
      {/* Animated background glow */}
      <motion.div
        className="absolute inset-0 z-0 opacity-50 blur-3xl"
        animate={{
          background: [
            "radial-gradient(circle at 20% 20%, rgba(79, 70, 229, 0.3), transparent 60%)",
            "radial-gradient(circle at 80% 80%, rgba(56, 189, 248, 0.3), transparent 60%)",
            "radial-gradient(circle at 50% 50%, rgba(147, 51, 234, 0.3), transparent 60%)",
          ],
        }}
        transition={{ duration: 10, repeat: Infinity, repeatType: "reverse" }}
      />

      {/* Main Title */}
      <motion.h1
        className="z-10 text-center text-4xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400 drop-shadow-lg mb-8"
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        Unified Health Dashboard
      </motion.h1>

      {/* Cards grid */}
      <motion.div
        className="z-10 grid grid-cols-1 lg:grid-cols-2 gap-8 w-full max-w-7xl"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <HarCard
          harConnected={harConnected}
          harActivity={harActivity}
          harConfidence={harConfidence}
          activityTimes={activityTimes}
        />

        <AudioCard
          coughCount={coughCount}
          audioStatus={audioStatus}
          audioResult={audioResult}
          isMonitoring={isMonitoring}
          onStart={startMonitoring}
          onStop={stopMonitoring}
        />
      </motion.div>

      {/* Bottom glowing bar */}
      <motion.div
        className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-indigo-500 via-cyan-400 to-indigo-500 opacity-50"
        animate={{ opacity: [0.3, 1, 0.3] }}
        transition={{ duration: 3, repeat: Infinity }}
      />

      <Modal show={showModal} onClose={() => setShowModal(false)} />
    </div>
  );
}

export default App;
