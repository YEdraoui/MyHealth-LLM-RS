// components/analysis/ImagePreview.tsx
'use client';

import { useState, useEffect } from 'react';
import { ZoomIn, ZoomOut, RotateCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ImagePreviewProps {
  file: File | null;
}

export function ImagePreview({ file }: ImagePreviewProps) {
  const [zoom, setZoom] = useState(100);
  const [rotation, setRotation] = useState(0);
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setImageUrl(null);
    }
  }, [file]);

  if (!file || !imageUrl) {
    return (
      <div className='bg-gray-800 rounded-lg p-6 flex items-center justify-center h-64 border-2 border-dashed border-gray-700'>
        <p className='text-gray-500'>No image uploaded</p>
      </div>
    );
  }

  return (
    <div className='bg-gray-800 rounded-lg p-4'>
      <div className='flex items-center justify-between mb-4'>
        <h3 className='text-lg font-semibold'> Image Preview</h3>
        <div className='flex gap-2'>
          <Button
            variant='outline'
            size='sm'
            onClick={() => setZoom(Math.max(25, zoom - 25))}
            disabled={zoom <= 25}
          >
            <ZoomOut className='h-4 w-4' />
          </Button>
          <Button
            variant='outline'
            size='sm'
            onClick={() => setZoom(Math.min(200, zoom + 25))}
            disabled={zoom >= 200}
          >
            <ZoomIn className='h-4 w-4' />
          </Button>
          <Button
            variant='outline'
            size='sm'
            onClick={() => setRotation((rotation + 90) % 360)}
          >
            <RotateCw className='h-4 w-4' />
          </Button>
          <span className='px-3 py-1 bg-gray-700 rounded text-sm'>{zoom}%</span>
        </div>
      </div>

      <div className='relative bg-gray-900 rounded-lg overflow-hidden' style={{ height: '400px' }}>
        <div className='absolute inset-0 flex items-center justify-center overflow-auto'>
          <img
            src={imageUrl}
            alt='Fundus Preview'
            className='transition-all duration-300'
            style={{
              transform: 'rotate(' + rotation + 'deg) scale(' + zoom / 100 + ')',
              maxWidth: '100%',
              maxHeight: '100%',
            }}
          />
        </div>
      </div>

      <div className='mt-2 text-sm text-gray-500'>
        {file.name}  {(file.size / 1024).toFixed(1)} KB
      </div>
    </div>
  );
}
