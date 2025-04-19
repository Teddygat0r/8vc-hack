import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Props = {
  content: string;
};
const fdsfdss = `# Hello Markdown 👋

This is a paragraph with **bold text** and a [link](https://example.com).

- List item 1
- List item 2
`;

const MarkdownRenderer: React.FC<Props> = ({ content }) => {
  return (
    <>
      <div className="w-full mt-8 bg-white bg-opacity-85 rounded-lg shadow-lg p-6 text-black">
        <div className="text-sm text-gray-400 mb-2">Response:</div>

        <div className="prose max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      </div>
    </>
  );
};

export default MarkdownRenderer;
