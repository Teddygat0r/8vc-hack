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
      <div className="prose prose-invert text-white">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </>
  );
};

export default MarkdownRenderer;
