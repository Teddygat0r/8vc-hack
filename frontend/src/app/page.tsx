"use client";

import { useEffect, useState, useRef } from "react";
import MarkdownRenderer from "./components/ReactMarkdown";
import SearchBar from "./components/SearchBar";

import { Doto } from "next/font/google";
const doto = Doto({
  weight: ["400", "700"],
  subsets: ["latin"],
  display: "swap",
});

export default function Home() {
  const [markdownContent, setMarkdownContent] = useState("");
  const [gradientAngle, setGradientAngle] = useState(0);
  const animationRef = useRef<number | null>(null);

  // Grab initial prompt from URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const prompt = params.get("prompt");
    if (prompt) setMarkdownContent(prompt);
  }, []);

  // Animate the gradient angle
  useEffect(() => {
    const animate = () => {
      setGradientAngle((a) => (a + 0.05) % 360);
      animationRef.current = requestAnimationFrame(animate);
    };
    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, []);

  const gradientStyle = {
    backgroundImage: `linear-gradient(${gradientAngle}deg, #4f46e5, #8b5cf6, #ec4899, #f97316)`,
    backgroundSize: "400% 400%",
    animation: "gradientShift 20s ease infinite",
  };

  return (
    <>
      {/* full‑screen fixed gradient wallpaper */}
      <div className="fixed inset-0 -z-10" style={gradientStyle} />

      {/* Vertically center title and search bar */}
      <main className="relative min-h-screen flex flex-col items-center justify-center">
        <div className="container mx-auto p-4 flex flex-col items-center">
          <h1 className={`${doto.className} text-6xl font-bold text-white mb-10`}>
            The Search Bar.
          </h1>
          <SearchBar setMarkdownText={setMarkdownContent} />
          {markdownContent && (
            <div className="w-full max-w-3xl mt-8">
              <MarkdownRenderer markdown={markdownContent} />
            </div>
          )}
        </div>
      </main>

      {/* global keyframes for the gradient animation */}
      <style jsx global>{`
        @keyframes gradientShift {
          0%   { background-position: 0% 50%; }
          50%  { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        @keyframes bounce-dot {
          0%, 100% { transform: translateY(0); }
          50%      { transform: translateY(-4px); }
        }
        .animate-bounce-dot { animation: bounce-dot 1s infinite; }
        .animate-bounce-dot-delay-1 { animation: bounce-dot 1s infinite; animation-delay: 0.2s; }
        .animate-bounce-dot-delay-2 { animation: bounce-dot 1s infinite; animation-delay: 0.4s; }
      `}</style>
    </>
  );
}