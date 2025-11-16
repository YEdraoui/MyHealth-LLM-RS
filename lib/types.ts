// lib/types.ts
// TypeScript types for the entire application

export interface AnalysisResult {
  severity: SeverityResult;
  conditions: ConditionResult[];
  report: string;
  gradcam?: string | null;
  processing_time?: ProcessingTime;
  timestamp?: string;
  metadata?: {
    filename: string;
    timestamp: string;
    report_type: string;
  };
}

export interface SeverityResult {
  level: 0 | 1 | 2 | 3 | 4;
  confidence: number;
  name: string;
  distribution?: number[];
}

export interface ConditionResult {
  name: string;
  probability: number;
  detected: boolean;
  threshold?: number;
}

export interface ProcessingTime {
  total: number;
  vision: number;
  vlm: number;
  breakdown?: {
    image_loading?: number;
    vision_model?: number;
    vlm_generation?: number;
    kb_retrieval?: number;
    report_formatting?: number;
  };
}

export type ReportType = 'brief' | 'comprehensive' | 'technical';

export interface HealthStatus {
  status: 'ok' | 'error';
  pipeline_ready: boolean;
  timestamp: string;
}
