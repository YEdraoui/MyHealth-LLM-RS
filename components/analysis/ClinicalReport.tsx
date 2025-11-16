// components/analysis/ClinicalReport.tsx
'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ClinicalReportProps {
  report: string;
}

export function ClinicalReport({ report }: ClinicalReportProps) {
  const [expanded, setExpanded] = useState(true);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(report);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Split report into sections
  const sections = report.split('\n\n').filter(s => s.trim());

  return (
    <div className='bg-gray-800 rounded-lg p-4'>
      <div className='flex items-center justify-between mb-4'>
        <h3 className='text-lg font-semibold flex items-center gap-2'>
          📋 Clinical Report
        </h3>
        <div className='flex gap-2'>
          <Button
            variant='outline'
            size='sm'
            onClick={handleCopy}
          >
            {copied ? <Check className='h-4 w-4' /> : <Copy className='h-4 w-4' />}
            {copied ? 'Copied!' : 'Copy'}
          </Button>
          <Button
            variant='outline'
            size='sm'
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? <ChevronUp className='h-4 w-4' /> : <ChevronDown className='h-4 w-4' />}
            {expanded ? 'Collapse' : 'Expand'}
          </Button>
        </div>
      </div>

      {expanded && (
        <div className='space-y-4 max-h-96 overflow-y-auto'>
          {sections.map((section, idx) => {
            const isHeader = section.includes('CLINICAL') || 
                           section.includes('KEY FINDINGS') || 
                           section.includes('RECOMMENDATIONS') ||
                           section.includes('FOLLOW-UP');
            
            return (
              <div key={idx} className={isHeader ? 'border-l-4 border-blue-500 pl-4' : ''}>
                <p className={
                  isHeader 
                    ? 'text-sm font-bold text-blue-400 mb-2' 
                    : 'text-sm text-gray-300 whitespace-pre-wrap'
                }>
                  {section}
                </p>
              </div>
            );
          })}
        </div>
      )}

      {!expanded && (
        <div className='text-sm text-gray-500 italic'>
          Click expand to view full report...
        </div>
      )}
    </div>
  );
}
