import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BlitzMate — Play Chess Against the Engine",
  description: "Play against BlitzMate, a strong and stateless classical chess engine built in Python.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
