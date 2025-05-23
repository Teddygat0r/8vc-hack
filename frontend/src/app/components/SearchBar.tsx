"use client";

import { useState, useEffect, useRef } from "react";

interface SearchBarProps {
  setMarkdownText: (text: string) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ setMarkdownText }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-focus on mount
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch("http://127.0.0.1:5000/api/process-prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: searchQuery }),
      });
      if (!res.ok) throw new Error("Network response was not ok");
      const data = await res.json();
      setMarkdownText(data.response || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-grow textarea
  const adjustTextareaHeight = (el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    el.style.height = Math.max(40, el.scrollHeight) + "px";
    el.style.height = `${el.scrollHeight}px`;
  };

  return (
    <div className="w-full max-w-3xl">
      {/* Search box */}
      <div className="w-full bg-white bg-opacity-85 rounded-full shadow-lg p-4">
        {isLoading && (
            <>
            <div className="fixed inset-0 bg-black opacity-30"></div>
            <div className="fixed inset-0 flex items-center text-2xl justify-center text-center text-black-400 gap-1">
              <span className="text-2xl">Loading</span>
              <span className="animate-bounce-dot">.</span>
              <span className="animate-bounce-dot-delay-1">.</span>
              <span className="animate-bounce-dot-delay-2">.</span>
            </div>
            </>
        )}
        <textarea
          ref={textareaRef}
          placeholder="Ask me anything..."
          className="w-full bg-white text-black p-3 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none overflow-hidden"
          value={searchQuery}
          onChange={e => {
            setSearchQuery(e.target.value);
            adjustTextareaHeight(e.target);
          }}
          onKeyDown={e => {
            if (e.key === "Enter" && !e.shiftKey && !isLoading) {
              e.preventDefault();
              handleSearch();
            }
          }}
          disabled={isLoading}
          rows={1}
          style={{ minHeight: "40px" }}
        />
        {error && (
          <div className="mt-2 text-red-400 text-sm">
            {error}
          </div>
        )}
      </div>

      
    </div>
  );
};

export default SearchBar;