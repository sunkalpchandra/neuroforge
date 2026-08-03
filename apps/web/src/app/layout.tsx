import type { Metadata, Viewport } from 'next';
import { IBM_Plex_Mono, Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const plexMono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-plex-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'NeuroForge — Neural Circuit CAD',
  description:
    'Design, simulate and visualise spiking neural circuits in the browser with GPU-accelerated biophysics.',
  applicationName: 'NeuroForge',
  keywords: [
    'neuroscience',
    'spiking neural network',
    'simulation',
    'WebGPU',
    'circuit design',
  ],
};

export const viewport: Viewport = {
  themeColor: '#07090B',
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${plexMono.variable}`}>
      <body className="bg-bg text-ink antialiased">{children}</body>
    </html>
  );
}
