// components/layout/Navbar.tsx
'use client';

import Link from 'next/link';
import { Activity, Home, Layers, History } from 'lucide-react';

export function Navbar() {
  return (
    <nav className='border-b border-gray-800 bg-gray-900'>
      <div className='container mx-auto px-4'>
        <div className='flex h-16 items-center justify-between'>
          {/* Logo */}
          <Link href='/' className='flex items-center space-x-2'>
            <Activity className='h-6 w-6 text-blue-500' />
            <span className='text-xl font-bold text-white'>
              Ophthalmology AI
            </span>
          </Link>

          {/* Navigation Links */}
          <div className='flex items-center space-x-6'>
            <Link
              href='/'
              className='flex items-center space-x-2 text-gray-300 hover:text-white transition-colors'
            >
              <Home className='h-4 w-4' />
              <span>Home</span>
            </Link>
            <Link
              href='/dashboard'
              className='flex items-center space-x-2 text-gray-300 hover:text-white transition-colors'
            >
              <Layers className='h-4 w-4' />
              <span>Dashboard</span>
            </Link>
            <Link
              href='/history'
              className='flex items-center space-x-2 text-gray-300 hover:text-white transition-colors'
            >
              <History className='h-4 w-4' />
              <span>History</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
