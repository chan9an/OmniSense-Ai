// src/components/AnimatedCounter.jsx
import { motion, AnimatePresence } from 'framer-motion';

export function AnimatedCounter({ value, className }) {
  return (
    <div className={`relative ${className}`} style={{ lineHeight: '1em' }}>
      <AnimatePresence>
        <motion.span
          key={value}
          initial={{ y: "100%", opacity: 0 }}
          animate={{ y: "0%", opacity: 1 }}
          exit={{ y: "-100%", opacity: 0 }}
          transition={{ duration: 0.3, ease: 'easeInOut' }}
          className="absolute inset-0"
        >
          {value}
        </motion.span>
      </AnimatePresence>
    </div>
  );
}