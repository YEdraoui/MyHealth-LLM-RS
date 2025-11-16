// lib/api.ts
// API client for Python backend communication

import { AnalysisResult, HealthStatus, ReportType } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class APIClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Check if Python backend is healthy
   */
  async healthCheck(): Promise<HealthStatus> {
    const response = await fetch(this.baseURL + '/api/health');
    if (!response.ok) {
      throw new Error('Health check failed');
    }
    return response.json();
  }

  /**
   * Analyze a fundus image
   */
  async analyzeImage(
    file: File,
    reportType: ReportType = 'brief',
    onProgress?: (progress: number) => void
  ): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('report_type', reportType);

    const response = await fetch(this.baseURL + '/api/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.details || error.error || 'Analysis failed');
    }

    return response.json();
  }

  /**
   * Batch analyze multiple images (Phase 4)
   */
  async batchAnalyze(files: File[]): Promise<AnalysisResult[]> {
    throw new Error('Batch analysis not implemented yet (Phase 4)');
  }
}

// Export singleton instance
export const apiClient = new APIClient();
