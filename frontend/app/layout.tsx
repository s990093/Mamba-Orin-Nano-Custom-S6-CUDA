import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Mamba2 Chat',
  description: 'Chat interface for Mamba2 MLX with comprehensive metrics',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
