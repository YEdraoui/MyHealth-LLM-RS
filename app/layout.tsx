import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Navbar } from '@/components/layout/Navbar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Ophthalmology AI - Clinical Decision Support',
  description: 'AI-powered diabetic retinopathy detection and analysis',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang='en' className='dark'>
      <body className={inter.className}>
        <div className='min-h-screen bg-gray-950 text-white'>
          <Navbar />
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
