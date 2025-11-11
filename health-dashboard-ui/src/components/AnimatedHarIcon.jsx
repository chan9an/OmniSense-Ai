// src/components/AnimatedHarIcon.jsx
import { motion, AnimatePresence } from 'framer-motion';

// --- Icon Definitions ---
const ICONS = {
  "Walking":    '<svg class="h-32 w-32 mx-auto text-cyan-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M11.25 4.5l7.5 7.5-7.5 7.5m-6-15l7.5 7.5-7.5 7.5" /></svg>',
  "Jogging":    '<svg class="h-32 w-32 mx-auto text-lime-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12.75 3.75v16.5M6.75 3.75v16.5M3 3.75h18M3 20.25h18" /></svg>',
  "Still":      '<svg class="h-32 w-32 mx-auto text-sky-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>',
  "Upstairs":   '<svg class="h-32 w-32 mx-auto text-emerald-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M4.5 15.75l7.5-7.5 7.5 7.5" /></svg>',
  "Downstairs": '<svg class="h-32 w-32 mx-auto text-purple-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" /></svg>',
  "Waiting...": '<svg class="h-32 w-32 mx-auto text-slate-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>'
};

export function AnimatedHarIcon({ activity }) {
  return (
    <div className="text-8xl my-6 text-slate-500 h-32 flex items-center justify-center">
      <AnimatePresence mode="wait">
        <motion.div
          key={activity}
          initial={{ opacity: 0, rotateY: -90 }}
          animate={{ opacity: 1, rotateY: 0 }}
          exit={{ opacity: 0, rotateY: 90 }}
          transition={{ duration: 0.3 }}
          dangerouslySetInnerHTML={{ __html: ICONS[activity] || ICONS["Waiting..."] }}
        />
      </AnimatePresence>
    </div>
  );
}