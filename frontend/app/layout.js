import "./globals.css";

export const metadata = {
  title: "RAG Enterprise — Document Intelligence",
  description:
    "Upload documents, ask questions, get evidence-grounded answers with source citations.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
