// components/shared/LoadingSpinner.tsx
export function LoadingSpinner({ message = 'Loading...' }: { message?: string }) {
  return (
    <div className='flex flex-col items-center justify-center p-12'>
      <div className='animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4'></div>
      <p className='text-gray-400'>{message}</p>
    </div>
  );
}
