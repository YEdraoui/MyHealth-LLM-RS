// stores/analysisStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AnalysisResult } from '@/lib/types';

interface AnalysisRecord extends AnalysisResult {
  id: string;
  timestamp: string;
  filename: string;
}

interface AnalysisStore {
  history: AnalysisRecord[];
  addAnalysis: (result: AnalysisResult, filename: string) => void;
  deleteAnalysis: (id: string) => void;
  clearHistory: () => void;
  getStats: () => {
    total: number;
    byLevel: Record<number, number>;
    averageConfidence: number;
  };
}

export const useAnalysisStore = create<AnalysisStore>()(
  persist(
    (set, get) => ({
      history: [],
      
      addAnalysis: (result, filename) => {
        const record: AnalysisRecord = {
          ...result,
          id: Date.now().toString(),
          timestamp: new Date().toISOString(),
          filename,
        };
        set((state) => ({
          history: [record, ...state.history].slice(0, 100), // Keep last 100
        }));
      },
      
      deleteAnalysis: (id) => {
        set((state) => ({
          history: state.history.filter((item) => item.id !== id),
        }));
      },
      
      clearHistory: () => set({ history: [] }),
      
      getStats: () => {
        const { history } = get();
        const byLevel: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
        let totalConfidence = 0;
        
        history.forEach((item) => {
          byLevel[item.severity.level]++;
          totalConfidence += item.severity.confidence;
        });
        
        return {
          total: history.length,
          byLevel,
          averageConfidence: history.length > 0 ? totalConfidence / history.length : 0,
        };
      },
    }),
    {
      name: 'analysis-storage',
    }
  )
);
