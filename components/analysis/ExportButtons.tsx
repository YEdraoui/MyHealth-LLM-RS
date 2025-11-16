// components/analysis/ExportButtons.tsx
'use client';

import { Download, FileJson, FileSpreadsheet, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { AnalysisResult } from '@/lib/types';

interface ExportButtonsProps {
  result: AnalysisResult;
}

export function ExportButtons({ result }: ExportButtonsProps) {
  const exportJSON = () => {
    const dataStr = JSON.stringify(result, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'analysis-result.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportCSV = () => {
    const headers = ['Condition', 'Probability', 'Detected'];
    const rows = result.conditions.map(c => [
      c.name,
      (c.probability * 100).toFixed(1) + '%',
      c.detected ? 'Yes' : 'No'
    ]);
    
    const csv = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'analysis-results.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportTXT = () => {
    const text = [
      'OPHTHALMOLOGY AI - ANALYSIS REPORT',
      '=' .repeat(50),
      '',
      'SEVERITY:',
      result.severity.name + ' (Level ' + result.severity.level + '/4)',
      'Confidence: ' + (result.severity.confidence * 100).toFixed(1) + '%',
      '',
      'DETECTED CONDITIONS:',
      ...result.conditions.map(c => 
        '- ' + c.name + ': ' + (c.probability * 100).toFixed(1) + '% ' + (c.detected ? '[DETECTED]' : '')
      ),
      '',
      'CLINICAL REPORT:',
      result.report,
      '',
      '=' .repeat(50),
      'Generated: ' + new Date().toLocaleString(),
    ].join('\n');
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'clinical-report.txt';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className='bg-gray-800 rounded-lg p-4'>
      <h3 className='text-lg font-semibold mb-4'>💾 Export Options</h3>
      
      <div className='grid grid-cols-3 gap-3'>
        <Button
          variant='outline'
          className='flex flex-col items-center gap-2 h-auto py-4'
          onClick={exportJSON}
        >
          <FileJson className='h-6 w-6' />
          <span className='text-sm'>JSON</span>
        </Button>
        
        <Button
          variant='outline'
          className='flex flex-col items-center gap-2 h-auto py-4'
          onClick={exportCSV}
        >
          <FileSpreadsheet className='h-6 w-6' />
          <span className='text-sm'>CSV</span>
        </Button>
        
        <Button
          variant='outline'
          className='flex flex-col items-center gap-2 h-auto py-4'
          onClick={exportTXT}
        >
          <FileText className='h-6 w-6' />
          <span className='text-sm'>TXT</span>
        </Button>
      </div>
    </div>
  );
}
