// components/analysis/ProcessingTimeline.tsx
'use client';

import { Clock, CheckCircle2, Loader2 } from 'lucide-react';

interface TimelineStep {
  name: string;
  duration: number;
  status: 'pending' | 'active' | 'completed';
}

interface ProcessingTimelineProps {
  isProcessing: boolean;
  currentStep?: string;
}

export function ProcessingTimeline({ isProcessing, currentStep }: ProcessingTimelineProps) {
  const steps: TimelineStep[] = [
    { name: 'Image Loading', duration: 2, status: isProcessing ? 'completed' : 'pending' },
    { name: 'Vision Model Analysis', duration: 4, status: isProcessing ? 'completed' : 'pending' },
    { name: 'VLM Report Generation', duration: 180, status: isProcessing ? 'active' : 'pending' },
    { name: 'Knowledge Base Retrieval', duration: 2, status: 'pending' },
    { name: 'Report Formatting', duration: 3, status: 'pending' },
  ];

  if (!isProcessing) return null;

  const totalTime = steps.reduce((sum, step) => sum + step.duration, 0);

  return (
    <div className='bg-gray-800 rounded-lg p-4 mb-4'>
      <h3 className='text-lg font-semibold mb-4 flex items-center gap-2'>
        <Clock className='h-5 w-5' />
        Processing Timeline
      </h3>

      <div className='space-y-3'>
        {steps.map((step, idx) => (
          <div key={idx} className='flex items-center gap-3'>
            <div className='flex-shrink-0'>
              {step.status === 'completed' ? (
                <CheckCircle2 className='h-5 w-5 text-green-500' />
              ) : step.status === 'active' ? (
                <Loader2 className='h-5 w-5 text-blue-500 animate-spin' />
              ) : (
                <Clock className='h-5 w-5 text-gray-500' />
              )}
            </div>

            <div className='flex-1'>
              <div className='flex items-center justify-between mb-1'>
                <span className={'text-sm font-medium ' + 
                  (step.status === 'completed' ? 'text-green-400' : 
                   step.status === 'active' ? 'text-blue-400' : 'text-gray-500')}>
                  {step.name}
                </span>
                <span className='text-xs text-gray-500'>{step.duration}s</span>
              </div>
              
              <div className='w-full bg-gray-700 rounded-full h-1.5'>
                <div
                  className={'h-1.5 rounded-full transition-all duration-500 ' +
                    (step.status === 'completed' ? 'bg-green-500 w-full' :
                     step.status === 'active' ? 'bg-blue-500 w-1/2' : 'w-0')}
                />
              </div>
            </div>

            <div className='text-xs text-gray-500 w-12 text-right'>
              {Math.round((step.duration / totalTime) * 100)}%
            </div>
          </div>
        ))}
      </div>

      <div className='mt-4 pt-4 border-t border-gray-700'>
        <div className='flex items-center justify-between text-sm'>
          <span className='text-gray-400'>Total Estimated Time</span>
          <span className='font-semibold'>{Math.floor(totalTime / 60)}m {totalTime % 60}s</span>
        </div>
      </div>
    </div>
  );
}
