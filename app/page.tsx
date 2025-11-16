// app/page.tsx
'use client';

import Link from 'next/link';
import { ArrowRight, Eye, Brain, Zap, History } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { AnalyticsCard } from '@/components/shared/AnalyticsCard';

export default function HomePage() {
  return (
    <div className='container mx-auto px-4 py-16'>
      {/* Hero Section */}
      <div className='text-center mb-16'>
        <h1 className='text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent'>
          Ophthalmology AI
        </h1>
        <p className='text-xl text-gray-400 mb-8'>
          Clinical Decision Support System for Diabetic Retinopathy Detection
        </p>
        <div className='flex gap-4 justify-center'>
          <Link href='/dashboard'>
            <Button size='lg' className='bg-blue-600 hover:bg-blue-700'>
              Start Analysis <ArrowRight className='ml-2 h-4 w-4' />
            </Button>
          </Link>
          <Link href='/history'>
            <Button size='lg' variant='outline'>
              <History className='mr-2 h-4 w-4' />
              View History
            </Button>
          </Link>
        </div>
      </div>

      {/* Analytics */}
      <div className='mb-16'>
        <AnalyticsCard />
      </div>

      {/* Features */}
      <div className='grid md:grid-cols-3 gap-8 mb-16'>
        <div className='bg-gray-900 border border-gray-800 rounded-lg p-6'>
          <Eye className='h-12 w-12 text-blue-500 mb-4' />
          <h3 className='text-xl font-semibold mb-2'>Vision Model</h3>
          <p className='text-gray-400'>
            ConvNeXt-Tiny with 83.5% AUROC detecting 13 conditions
          </p>
        </div>

        <div className='bg-gray-900 border border-gray-800 rounded-lg p-6'>
          <Brain className='h-12 w-12 text-purple-500 mb-4' />
          <h3 className='text-xl font-semibold mb-2'>AI Reports</h3>
          <p className='text-gray-400'>
            Llama 3.2-Vision 11B generates clinical reports
          </p>
        </div>

        <div className='bg-gray-900 border border-gray-800 rounded-lg p-6'>
          <Zap className='h-12 w-12 text-green-500 mb-4' />
          <h3 className='text-xl font-semibold mb-2'>Fast Analysis</h3>
          <p className='text-gray-400'>
            Complete analysis in ~3 minutes with detailed insights
          </p>
        </div>
      </div>

      {/* Stats */}
      <div className='bg-gradient-to-r from-blue-900/20 to-purple-900/20 border border-blue-800/30 rounded-lg p-8'>
        <div className='grid md:grid-cols-4 gap-8 text-center'>
          <div>
            <div className='text-3xl font-bold text-blue-400'>83.5%</div>
            <div className='text-gray-400'>AUROC Score</div>
          </div>
          <div>
            <div className='text-3xl font-bold text-purple-400'>13</div>
            <div className='text-gray-400'>Conditions Detected</div>
          </div>
          <div>
            <div className='text-3xl font-bold text-green-400'>60K+</div>
            <div className='text-gray-400'>Training Images</div>
          </div>
          <div>
            <div className='text-3xl font-bold text-orange-400'>5</div>
            <div className='text-gray-400'>Severity Levels</div>
          </div>
        </div>
      </div>
    </div>
  );
}
