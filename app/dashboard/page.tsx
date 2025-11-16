// app/dashboard/page.tsx
'use client';

import { useState } from 'react';
import { Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { apiClient } from '@/lib/api';
import type { AnalysisResult, ReportType } from '@/lib/types';
import { useAnalysisStore } from '@/stores/analysisStore';

// Import all components
import { SeverityGauge } from '@/components/analysis/SeverityGauge';
import { PredictionsTable } from '@/components/analysis/PredictionsTable';
import { ImagePreview } from '@/components/analysis/ImagePreview';
import { ProcessingTimeline } from '@/components/analysis/ProcessingTimeline';
import { ClinicalReport } from '@/components/analysis/ClinicalReport';
import { ExportButtons } from '@/components/analysis/ExportButtons';

export default function DashboardPage() {
  const [file, setFile] = useState<File | null>(null);
  const [reportType, setReportType] = useState<ReportType>('brief');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const addAnalysis = useAnalysisStore((state) => state.addAnalysis);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const analysisResult = await apiClient.analyzeImage(file, reportType);
      console.log('Raw result:', analysisResult);
      
      // Validate result has required fields
      if (!analysisResult.severity || !analysisResult.conditions || !analysisResult.report) {
        throw new Error('Invalid response from server');
      }
      
      setResult(analysisResult);
      
      // Save to history
      addAnalysis(analysisResult, file.name);
    } catch (err: unknown) {
      console.error('Analysis error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className='container mx-auto px-4 py-8'>
      <h1 className='text-3xl font-bold mb-8'>Analysis Dashboard</h1>

      <div className='grid lg:grid-cols-3 gap-6'>
        {/* Left Column - Upload & Image Preview */}
        <div className='space-y-6'>
          <Card className='p-6 bg-gray-900 border-gray-800'>
            <h2 className='text-xl font-semibold mb-4'>Upload Image</h2>
            
            <div className='space-y-4'>
              <div className='border-2 border-dashed border-gray-700 rounded-lg p-8 text-center'>
                <Upload className='h-12 w-12 mx-auto mb-4 text-gray-500' />
                <input
                  type='file'
                  accept='image/*'
                  onChange={handleFileChange}
                  className='hidden'
                  id='file-upload'
                />
                <label htmlFor='file-upload'>
                  <Button variant='outline' className='cursor-pointer' asChild>
                    <span>Select Image</span>
                  </Button>
                </label>
                {file && (
                  <p className='mt-4 text-sm text-gray-400'>
                    Selected: {file.name}
                  </p>
                )}
              </div>

              <div>
                <label className='block text-sm font-medium mb-2'>Report Type</label>
                <select
                  value={reportType}
                  onChange={(e) => setReportType(e.target.value as ReportType)}
                  className='w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white'
                >
                  <option value='brief'>Brief</option>
                  <option value='comprehensive'>Comprehensive</option>
                  <option value='technical'>Technical</option>
                </select>
              </div>

              <Button
                onClick={handleAnalyze}
                disabled={!file || loading}
                className='w-full bg-blue-600 hover:bg-blue-700'
              >
                {loading ? 'Analyzing...' : 'Analyze Image'}
              </Button>

              {error && (
                <div className='bg-red-900/20 border border-red-800 rounded-lg p-4 text-red-400'>
                  {error}
                </div>
              )}
            </div>
          </Card>

          {file && <ImagePreview file={file} />}
        </div>

        {/* Middle & Right Columns - Results */}
        <div className='lg:col-span-2 space-y-6'>
          <ProcessingTimeline isProcessing={loading} />

          {result && result.severity && result.conditions && (
            <>
              <SeverityGauge
                level={result.severity.level}
                confidence={result.severity.confidence}
                name={result.severity.name}
              />

              <PredictionsTable conditions={result.conditions} />

              <ClinicalReport report={result.report} />

              <ExportButtons result={result} />
            </>
          )}

          {!loading && !result && (
            <Card className='p-12 bg-gray-900 border-gray-800'>
              <div className='text-center text-gray-500'>
                <Upload className='h-16 w-16 mx-auto mb-4 opacity-50' />
                <p className='text-lg'>Upload and analyze an image to see results</p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
