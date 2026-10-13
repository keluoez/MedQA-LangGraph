/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      keyframes: {
        breathing: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.45" },
        },
        flow: {
          to: { backgroundPosition: "24px 0" },
        },
        fadeUp: {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        breathing: "breathing 1.6s ease-in-out infinite",
        flow: "flow 0.7s linear infinite",
        "fade-up": "fadeUp 0.25s ease-out",
      },
    },
  },
  plugins: [],
};
