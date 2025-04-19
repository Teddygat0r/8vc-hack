"use client";

import { useEffect, useState } from "react";
import MarkdownRenderer from "./components/ReactMarkdown";
import SearchBar from "./components/SearchBar";

export default function Home() {
  const [markdownContent, setMarkdownContent] = useState("");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const prompt = params.get("prompt");
    if (prompt) {
      setMarkdownContent(prompt);
    }
  }, []);

  return (
    <main className="bg-gray-900 min-h-screen flex">
      <div className="container mx-auto p-4 flex flex-col items-center justify-center">
        <div className="mx-[20%] my-[10%] bg-gray-800 rounded-lg shadow-lg p-8 flex w-full text-white">
          <MarkdownRenderer content={markdownContent}/>
        </div>
        <SearchBar setMarkdownText={setMarkdownContent}></SearchBar>
      </div>
    </main>
  );
}
