export function Modal({ show, onClose }) {
  if (!show) return null;
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-slate-800 p-8 rounded-2xl shadow-xl w-96 text-center">
        <h2 className="text-2xl text-yellow-400 font-bold mb-2">Health Tip</h2>
        <p className="text-slate-300 mb-4">You’re coughing quite a lot. Try these tips:</p>
        <ul className="text-left text-slate-200 text-sm mb-4 list-disc list-inside">
          <li>Drink warm fluids and rest your throat.</li>
          <li>Keep your room humidified.</li>
          <li>Consult a doctor if persistent.</li>
        </ul>
        <button
          onClick={onClose}
          className="bg-indigo-600 hover:bg-indigo-700 px-6 py-2 rounded-full text-white font-semibold"
        >
          Got it
        </button>
      </div>
    </div>
  );
}
