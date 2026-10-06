/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0D0D0D', // ONYX BLACK
        surface: '#1C1C1E', // CHARCOAL
        surfaceBorder: '#3A2F2A', // ESPRESSO
        textMain: '#F5F5F5', // Soft white for contrast
        textMuted: '#6B6965', // SLATE GREY
        accent: '#B08D57' // ANTIQUE GOLD
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['"Inter"', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
