// components/analysis/PredictionsTable.tsx
'use client';

import { useState } from 'react';
import { Search } from 'lucide-react';
import type { ConditionResult } from '@/lib/types';

interface PredictionsTableProps {
  conditions: ConditionResult[];
}

export function PredictionsTable({ conditions }: PredictionsTableProps) {
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState<'name' | 'probability'>('probability');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Filter and sort conditions
  const filteredConditions = conditions
    .filter((c) => c.name.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      const aVal = sortBy === 'name' ? a.name : a.probability;
      const bVal = sortBy === 'name' ? b.name : b.probability;
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortOrder === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortOrder === 'asc' ? Number(aVal) - Number(bVal) : Number(bVal) - Number(aVal);
    });

  const getStatusBadge = (probability: number) => {
    if (probability >= 0.5) return { label: '✅ YES', color: 'bg-red-500/20 text-red-400 border-red-500' };
    if (probability >= 0.2) return { label: '⚠️ MAYBE', color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500' };
    return { label: '❌ NO', color: 'bg-gray-500/20 text-gray-400 border-gray-500' };
  };

  return (
    <div className='bg-gray-800 rounded-lg p-4'>
      <div className='flex items-center justify-between mb-4'>
        <h3 className='text-lg font-semibold'>🔍 Detected Conditions</h3>
        <div className='relative'>
          <Search className='absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500' />
          <input
            type='text'
            placeholder='Search...'
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className='pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-sm focus:outline-none focus:border-blue-500'
          />
        </div>
      </div>

      <div className='overflow-x-auto'>
        <table className='w-full'>
          <thead>
            <tr className='border-b border-gray-700'>
              <th
                className='text-left py-2 px-3 text-sm font-medium text-gray-400 cursor-pointer hover:text-white'
                onClick={() => {
                  setSortBy('name');
                  setSortOrder(sortBy === 'name' && sortOrder === 'asc' ? 'desc' : 'asc');
                }}
              >
                Condition {sortBy === 'name' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th
                className='text-left py-2 px-3 text-sm font-medium text-gray-400 cursor-pointer hover:text-white'
                onClick={() => {
                  setSortBy('probability');
                  setSortOrder(sortBy === 'probability' && sortOrder === 'asc' ? 'desc' : 'asc');
                }}
              >
                Confidence {sortBy === 'probability' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th className='text-left py-2 px-3 text-sm font-medium text-gray-400'>Visual</th>
              <th className='text-left py-2 px-3 text-sm font-medium text-gray-400'>Status</th>
            </tr>
          </thead>
          <tbody>
            {filteredConditions.map((condition, idx) => {
              const status = getStatusBadge(condition.probability);
              const percentage = Math.round(condition.probability * 100);
              
              return (
                <tr key={idx} className='border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors'>
                  <td className='py-3 px-3 text-sm font-medium'>
                    {condition.name}
                  </td>
                  <td className='py-3 px-3 text-sm text-gray-300'>
                    {percentage}%
                  </td>
                  <td className='py-3 px-3'>
                    <div className='w-full bg-gray-700 rounded-full h-2'>
                      <div
                        className={'h-2 rounded-full transition-all duration-500 ' + 
                          (condition.detected ? 'bg-red-500' : 'bg-gray-500')}
                        style={{ width: percentage + '%' }}
                      />
                    </div>
                  </td>
                  <td className='py-3 px-3'>
                    <span className={'px-2 py-1 rounded text-xs font-medium border ' + status.color}>
                      {status.label}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className='mt-4 text-sm text-gray-500'>
        Showing {filteredConditions.length} of {conditions.length} conditions
      </div>
    </div>
  );
}
