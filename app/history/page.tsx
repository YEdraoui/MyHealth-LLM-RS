// app/history/page.tsx
'use client';

import { useState } from 'react';
import { useAnalysisStore } from '@/stores/analysisStore';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Trash2, Search } from 'lucide-react';
import { format } from 'date-fns';

export default function HistoryPage() {
  const { history, deleteAnalysis, clearHistory } = useAnalysisStore();
  const [search, setSearch] = useState('');
  const [filterLevel, setFilterLevel] = useState<number | 'all'>('all');

  const filteredHistory = history.filter((item) => {
    const matchesSearch = item.filename.toLowerCase().includes(search.toLowerCase());
    const matchesFilter = filterLevel === 'all' || item.severity.level === filterLevel;
    return matchesSearch && matchesFilter;
  });

  const getSeverityColor = (level: number) => {
    const colors = ['text-green-400', 'text-yellow-400', 'text-orange-400', 'text-red-400', 'text-red-600'];
    return colors[level];
  };

  return (
    <div className='container mx-auto px-4 py-8'>
      <div className='flex items-center justify-between mb-8'>
        <h1 className='text-3xl font-bold'>Analysis History</h1>
        <Button
          variant='outline'
          onClick={clearHistory}
          disabled={history.length === 0}
          className='text-red-400 border-red-400 hover:bg-red-400/10'
        >
          <Trash2 className='h-4 w-4 mr-2' />
          Clear All
        </Button>
      </div>

      <Card className='p-4 bg-gray-900 border-gray-800 mb-6'>
        <div className='flex gap-4'>
          <div className='flex-1'>
            <div className='relative'>
              <Search className='absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500' />
              <input
                type='text'
                placeholder='Search by filename...'
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className='w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-blue-500'
              />
            </div>
          </div>
          
          <select
            value={filterLevel}
            onChange={(e) => setFilterLevel(e.target.value === 'all' ? 'all' : Number(e.target.value))}
            className='bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm text-white'
          >
            <option value='all'>All Levels</option>
            <option value='0'>No DR</option>
            <option value='1'>Mild</option>
            <option value='2'>Moderate</option>
            <option value='3'>Severe</option>
            <option value='4'>Proliferative</option>
          </select>
        </div>
      </Card>

      {filteredHistory.length === 0 ? (
        <Card className='p-12 bg-gray-900 border-gray-800'>
          <div className='text-center text-gray-500'>
            <p className='text-lg'>No analysis history found</p>
            <p className='text-sm mt-2'>Analyze some images to see them here</p>
          </div>
        </Card>
      ) : (
        <div className='grid gap-4'>
          {filteredHistory.map((item) => (
            <Card key={item.id} className='p-6 bg-gray-900 border-gray-800 hover:border-gray-700 transition-colors'>
              <div className='flex items-start justify-between'>
                <div className='flex-1'>
                  <div className='flex items-center gap-3 mb-2'>
                    <h3 className='text-lg font-semibold'>{item.filename}</h3>
                    <span className={'px-3 py-1 rounded-full text-xs font-medium ' + getSeverityColor(item.severity.level)}>
                      {item.severity.name}
                    </span>
                  </div>
                  
                  <div className='grid grid-cols-3 gap-4 text-sm text-gray-400 mb-3'>
                    <div>
                      <span className='text-gray-500'>Date:</span>{' '}
                      {format(new Date(item.timestamp), 'PPpp')}
                    </div>
                    <div>
                      <span className='text-gray-500'>Confidence:</span>{' '}
                      {(item.severity.confidence * 100).toFixed(1)}%
                    </div>
                    <div>
                      <span className='text-gray-500'>Conditions:</span>{' '}
                      {item.conditions.filter(c => c.detected).length} detected
                    </div>
                  </div>

                  <div className='flex flex-wrap gap-2'>
                    {item.conditions
                      .filter(c => c.detected)
                      .slice(0, 5)
                      .map((condition, idx) => (
                        <span
                          key={idx}
                          className='px-2 py-1 bg-red-500/20 text-red-400 border border-red-500/30 rounded text-xs'
                        >
                          {condition.name}
                        </span>
                      ))}
                    {item.conditions.filter(c => c.detected).length > 5 && (
                      <span className='px-2 py-1 bg-gray-700 text-gray-400 rounded text-xs'>
                        +{item.conditions.filter(c => c.detected).length - 5} more
                      </span>
                    )}
                  </div>
                </div>

                <Button
                  variant='outline'
                  size='sm'
                  onClick={() => deleteAnalysis(item.id)}
                  className='text-red-400 border-red-400 hover:bg-red-400/10'
                >
                  <Trash2 className='h-4 w-4' />
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {history.length > 0 && (
        <Card className='p-4 bg-gray-900 border-gray-800 mt-6'>
          <div className='text-sm text-gray-400'>
            Showing {filteredHistory.length} of {history.length} total analyses
          </div>
        </Card>
      )}
    </div>
  );
}
