// components/analysis/SeverityGauge.tsx
'use client';

import { useEffect, useState } from 'react';

interface SeverityGaugeProps {
  level: 0 | 1 | 2 | 3 | 4;
  confidence: number;
  name: string;
}

export function SeverityGauge({ level, confidence, name }: SeverityGaugeProps) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Animate progress on mount
    const timer = setTimeout(() => setProgress(level / 4 * 100), 100);
    return () => clearTimeout(timer);
  }, [level]);

  // Color mapping for severity levels
  const colors = {
    0: { bg: 'bg-green-500', text: 'text-green-400', border: 'border-green-500', emoji: '😊' },
    1: { bg: 'bg-yellow-500', text: 'text-yellow-400', border: 'border-yellow-500', emoji: '⚠️' },
    2: { bg: 'bg-orange-500', text: 'text-orange-400', border: 'border-orange-500', emoji: '🔴' },
    3: { bg: 'bg-red-500', text: 'text-red-400', border: 'border-red-500', emoji: '🚨' },
    4: { bg: 'bg-red-900', text: 'text-red-300', border: 'border-red-900', emoji: '🆘' },
  };

  const color = colors[level];
  const percentage = Math.round(confidence * 100);

  return (
    <div className='bg-gray-800 rounded-lg p-6 border-2 border-gray-700'>
      <h3 className='text-lg font-semibold mb-4'>🎯 Severity Assessment</h3>
      
      <div className='flex items-center justify-between mb-6'>
        {/* Circular Progress */}
        <div className='relative w-32 h-32'>
          <svg className='w-32 h-32 transform -rotate-90'>
            {/* Background circle */}
            <circle
              cx='64'
              cy='64'
              r='56'
              stroke='currentColor'
              strokeWidth='8'
              fill='none'
              className='text-gray-700'
            />
            {/* Progress circle */}
            <circle
              cx='64'
              cy='64'
              r='56'
              stroke='currentColor'
              strokeWidth='8'
              fill='none'
              strokeDasharray={2 * Math.PI * 56}
              strokeDashoffset={2 * Math.PI * 56 * (1 - progress / 100)}
              className={'transition-all duration-1000 ease-out ' + color.bg.replace('bg-', 'text-')}
              strokeLinecap='round'
            />
          </svg>
          {/* Center text */}
          <div className='absolute inset-0 flex flex-col items-center justify-center'>
            <span className='text-3xl'>{color.emoji}</span>
            <span className={'text-2xl font-bold ' + color.text}>{level}</span>
            <span className='text-xs text-gray-400'>/4</span>
          </div>
        </div>

        {/* Details */}
        <div className='flex-1 ml-6'>
          <h4 className={'text-2xl font-bold mb-2 ' + color.text}>{name}</h4>
          <p className='text-sm text-gray-400 mb-2'>Level {level} of 4</p>
          <div className='flex items-center gap-2'>
            <div className='flex-1 bg-gray-700 rounded-full h-2'>
              <div
                className={'h-2 rounded-full transition-all duration-1000 ' + color.bg}
                style={{ width: percentage + '%' }}
              />
            </div>
            <span className='text-sm font-semibold text-gray-300'>{percentage}%</span>
          </div>
          <p className='text-xs text-gray-500 mt-2'>Confidence Score</p>
        </div>
      </div>

      {/* Severity Level Indicators */}
      <div className='flex justify-between items-center gap-2'>
        {[
          { level: 0, label: 'None', emoji: '😊' },
          { level: 1, label: 'Mild', emoji: '⚠️' },
          { level: 2, label: 'Moderate', emoji: '🔴' },
          { level: 3, label: 'Severe', emoji: '🚨' },
          { level: 4, label: 'Prolif.', emoji: '🆘' },
        ].map((item) => (
          <div
            key={item.level}
            className={'flex flex-col items-center p-2 rounded-lg flex-1 ' + 
              (level === item.level ? 'bg-gray-700 border-2 ' + color.border : 'bg-gray-800/50')}
          >
            <span className='text-xl'>{item.emoji}</span>
            <span className={'text-xs font-medium ' + (level === item.level ? color.text : 'text-gray-500')}>
              {item.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
