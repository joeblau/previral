import type { Metadata } from "next";
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "./globals.css";

export const metadata: Metadata = {
  title: "Previral — See your video differently",
  description:
    "Explore how video, sound, and language relate to predicted brain activity. Previral is a native macOS app powered by Meta’s TRIBE v2 and Apple Core ML.",
  openGraph: {
    title: "Previral — See your video differently",
    description:
      "A new lens on video. Explore predicted brain activity, frame by frame, on your Mac.",
    type: "website",
  },
  twitter: { card: "summary", title: "Previral — See your video differently" },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
