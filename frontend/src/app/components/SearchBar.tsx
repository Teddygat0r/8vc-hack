"use client"

import { useState } from "react";

const SearchBar: React.FC<{ setMarkdownText: (text: string) => void }> = ({
  setMarkdownText,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    setIsLoading(true);
    setError(null);

    try {
        const response = await fetch("http://127.0.0.1:5000/api/add-hash", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: searchQuery }),
        });

        if (!response.ok) {
            throw new Error("Network response was not ok");
        }

        const data = await response.json();
        console.log(data);
        setMarkdownText(data.result || "");
    } catch (error) {
        setError(error instanceof Error ? error.message : "An error occurred");
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-gray-900 p-3 shadow-md mx-[20%] text-white">
      <textarea
        placeholder="Type your message..."
        className="h-[2.5rem] max-h-[6.75rem] w-full bg-gray-800 text-white p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none overflow-auto m-0"
        value={searchQuery}
        onChange={(e) => {
          setSearchQuery(e.target.value);
          const textarea = e.target;
          textarea.style.height = "auto";
          textarea.style.height = Math.max(0, textarea.scrollHeight - 40) + "px";
          textarea.style.height = `${Math.min(textarea.scrollHeight, 108)}px`;
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !isLoading) {
            e.preventDefault();
            handleSearch();
          }
        }}
        disabled={isLoading}
      />
    </div>
  );
};

export default SearchBar;
