// components/shared/AnalyticsCard.tsx
'use client';

import { useAnalysisStore } from '@/stores/analysisStore';
import { Card } from '@/components/ui/card';
import { BarChart, Activity, TrendingUp } from 'lucide-react';

export function AnalyticsCard() {
  const { history, getStats } = useAnalysisStore();
  const stats = getStats();

  if (history.length === 0) return null;

  const levelNames = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];
  const levelColors = ['bg-green-500', 'bg-yellow-500', 'bg-orange-500', 'bg-red-500', 'bg-red-900'];

  return (
    <Card className='p-6 bg-gray-900 border-gray-800 mb-6'>
      <h3 className='text-lg font-semibold mb-4 flex items-center gap-2'>
        <Activity className='h-5 w-5' />
        Analytics Overview
      </h3>

      <div className='grid grid-cols-3 gap-6 mb-6'>
        <div className='text-center'>
          <div className='text-3xl font-bold text-blue-400'>{stats.total}</div>
          <div className='text-sm text-gray-500'>Total Analyses</div>
        </div>
        <div className='text-center'>
          <div className='text-3xl font-bold text-purple-400'>
            {(stats.averageConfidence * 100).toFixed(1)}%
          </div>
          <div className='text-sm text-gray-500'>Avg Confidence</div>
        </div>
        <div className='text-center'>
          <div className='text-3xl font-bold text-green-400'>
            {stats.byLevel[0]}
          </div>
          <div className='text-sm text-gray-500'>Healthy Cases</div>
        </div>
      </div>

      {/* Distribution Chart */}
      <div className='space-y-2'>
        <h4 className='text-sm font-medium text-gray-400 mb-3'>Severity Distribution</h4>
        {[0, 1, 2, 3, 4].map((level) => {
          const count = stats.byLevel[level];
          const percentage = stats.total > 0 ? (count / stats.total) * 100 : 0;
          
          return (
            <div key={level} className='flex items-center gap-3'>
              <div className='w-24 text-sm text-gray-400'>{levelNames[level]}</div>
              <div className='flex-1 bg-gray-800 rounded-full h-6 overflow-hidden'>
                <div
                  className={'h-full flex items-center justify-end pr-2 text-xs font-medium text-white transition-all duration-500 ' + levelColors[level]}
                  style={{ width: percentage + '%' }}
                >
                  {percentage > 10 && count + ' (' + percentage.toFixed(0) + '%)'}
                </div>
              </div>
              {percentage <= 10 && (
                <div className='w-16 text-sm text-gray-400'>{count}</div>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}
