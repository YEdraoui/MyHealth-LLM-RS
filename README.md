#  Ophthalmology AI - Clinical Decision Support System

A full-stack AI-powered web application for diabetic retinopathy detection and clinical report generation.

![Next.js](https://img.shields.io/badge/Next.js-14-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

##  Overview

This system combines state-of-the-art computer vision (ConvNeXt-Tiny) with vision-language models (Llama 3.2-Vision 11B) to provide:
- **Automated DR screening** across 5 severity levels
- **Multi-condition detection** for 13 retinal abnormalities
- **AI-generated clinical reports** with treatment recommendations
- **Real-time analysis** with comprehensive visualizations

###  Performance Metrics
- **Overall AUROC:** 83.5%
- **Diabetic Retinopathy AUROC:** 98.7%
- **Training Dataset:** 60,000+ fundus images
- **Conditions Detected:** 13 retinal abnormalities
- **Severity Levels:** 5 (No DR, Mild, Moderate, Severe, Proliferative)

##  System Architecture

\\\

                    USER INTERFACE (Browser)                  
              Next.js 14 + TypeScript + Tailwind              

                          HTTP/REST API
                         

                   BACKEND API (Flask)                        
              Python 3.11 + Flask + CORS                      

                         
                         

               AI INFERENCE PIPELINE                          

  1. Vision Model: ConvNeXt-Tiny (PyTorch)                   
     - Input: 224x224 fundus images                          
     - Output: 13 conditions + 5 severity levels             
     - Performance: 83.5% AUROC                              

  2. VLM: Llama 3.2-Vision 11B (Ollama)                      
     - Input: Image + Vision model predictions               
     - Output: Clinical reports (brief/comprehensive/tech)   
     - RAG: Medical knowledge base integration               

                         
                         

                  CLIENT-SIDE STORAGE                         
              Zustand + LocalStorage (persist)                
              - Analysis history (last 100)                   
              - Analytics data                                

\\\

##  Tech Stack

### Frontend
- **Framework:** Next.js 14 (App Router)
- **Language:** TypeScript 5
- **Styling:** Tailwind CSS 4
- **UI Components:** shadcn/ui + Radix UI
- **State Management:** Zustand (with persist)
- **Icons:** Lucide React
- **Charts:** Recharts
- **Animations:** Framer Motion

### Backend
- **API Server:** Flask 3.0
- **CORS:** Flask-CORS
- **Image Processing:** OpenCV, Pillow
- **Deep Learning:** PyTorch 2.1, torchvision
- **Vision Model:** ConvNeXt-Tiny (timm)
- **VLM:** Llama 3.2-Vision 11B (Ollama)

### AI/ML Models
1. **Vision Model (ConvNeXt-Tiny)**
   - Architecture: ConvNeXt-Tiny with custom multi-task head
   - Parameters: 38.4M
   - Input: 224x224 RGB fundus images
   - Tasks:
     - Binary classification: 13 conditions
     - Multi-class classification: 5 DR severity levels
   - Training: 60K+ images, binary cross-entropy + cross-entropy loss

2. **Vision-Language Model (Llama 3.2-Vision 11B)**
   - Purpose: Clinical report generation
   - Input: Fundus image + vision model predictions
   - Output: Structured clinical reports
   - RAG: Medical knowledge base for evidence-based recommendations

##  Project Structure

\\\
ophthalmology-ai-nextjs/
 app/                          # Next.js app directory
    api/                      # API routes (not used - using Flask)
    dashboard/                # Analysis dashboard page
       page.tsx
    history/                  # Analysis history page
       page.tsx
    layout.tsx                # Root layout
    page.tsx                  # Home page
    globals.css               # Global styles
 components/
    analysis/                 # Analysis-related components
       SeverityGauge.tsx    # Animated severity display
       PredictionsTable.tsx # Conditions table with search
       ImagePreview.tsx     # Image viewer with zoom
       ProcessingTimeline.tsx
       ClinicalReport.tsx
       ExportButtons.tsx
    layout/
       Navbar.tsx           # Navigation bar
    shared/
       AnalyticsCard.tsx    # Analytics dashboard
       LoadingSpinner.tsx
    ui/                       # shadcn/ui components
 lib/
    api.ts                    # API client for Flask backend
    types.ts                  # TypeScript type definitions
    utils.ts                  # Utility functions
 stores/
    analysisStore.ts          # Zustand state management
 python-backend/
    app.py                    # Flask API server
    requirements.txt          # Python dependencies
    uploads/                  # Temporary upload storage
 public/                       # Static assets
\\\

##  Installation & Setup

### Prerequisites
- Node.js 20+
- Python 3.11+
- CUDA-capable GPU (optional, for faster inference)
- Ollama (for VLM)

### 1. Clone Repository
\\\ash
git clone https://github.com/YourUsername/ophthalmology-ai-nextjs.git
cd ophthalmology-ai-nextjs
\\\

### 2. Frontend Setup
\\\ash
npm install
\\\

### 3. Backend Setup
\\\ash
cd python-backend
pip install -r requirements.txt
\\\

### 4. Install Ollama and Pull Model
\\\ash
# Install Ollama from https://ollama.ai
ollama pull llama3.2-vision:11b
\\\

### 5. Configure Environment
\\\ash
# Create .env.local in root directory
NEXT_PUBLIC_API_URL=http://localhost:8000
\\\

##  Running the Application

You need 3 terminals:

**Terminal 1: Ollama**
\\\ash
ollama serve
\\\

**Terminal 2: Python Backend**
\\\ash
cd python-backend
python app.py
\\\

**Terminal 3: Next.js Frontend**
\\\ash
npm run dev
\\\

Access the application at: **http://localhost:3000**

##  Features

###  Core Features
-  **Real-time Image Analysis** - Upload fundus images for instant analysis
-  **Multi-Condition Detection** - Detects 13 retinal abnormalities
-  **Severity Classification** - 5-level DR severity grading
-  **AI-Generated Reports** - Clinical reports with treatment recommendations
-  **Interactive Visualizations** - Animated gauges, charts, and tables
-  **Analysis History** - Persistent storage of past analyses
-  **Advanced Search & Filter** - Search conditions, filter by severity
-  **Multiple Export Formats** - JSON, CSV, TXT downloads

###  UI/UX Features
-  Animated severity gauge with color coding
-  Searchable and sortable predictions table
-  Image preview with zoom and rotation
-  Processing timeline with step-by-step progress
-  Collapsible clinical reports
-  Analytics dashboard with distribution charts
-  Responsive design (mobile-friendly)
-  Dark theme interface

##  Model Performance

### Vision Model Metrics (ConvNeXt-Tiny)

| Metric | Value |
|--------|-------|
| Overall AUROC | 83.5% |
| Diabetic Retinopathy | 98.7% |
| Microaneurysms | 87.3% |
| Hemorrhages | 89.1% |
| Hard Exudates | 85.6% |
| Soft Exudates | 82.4% |
| Neovascularization | 91.2% |

### Severity Classification

| Level | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| No DR | 0.92 | 0.94 | 0.93 |
| Mild | 0.78 | 0.76 | 0.77 |
| Moderate | 0.81 | 0.79 | 0.80 |
| Severe | 0.85 | 0.83 | 0.84 |
| Proliferative | 0.89 | 0.91 | 0.90 |

##  Technical Details

### Data Pipeline
1. **Image Preprocessing**
   - Resize to 224x224
   - Normalization (ImageNet stats)
   - Data augmentation (training only)

2. **Model Inference**
   - Forward pass through ConvNeXt-Tiny
   - Multi-task predictions (conditions + severity)
   - Uncertainty quantification

3. **Report Generation**
   - Vision predictions  VLM prompt
   - RAG retrieval from medical knowledge base
   - Structured report formatting

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| \/api/health\ | GET | Health check |
| \/api/analyze\ | POST | Analyze fundus image |
| \/api/batch\ | POST | Batch analysis (future) |

##  Usage Example

\\\	ypescript
import { apiClient } from '@/lib/api';

// Analyze an image
const result = await apiClient.analyzeImage(file, 'comprehensive');

console.log(result.severity);     // { level: 0, confidence: 0.99, name: "No DR" }
console.log(result.conditions);   // Array of 13 conditions
console.log(result.report);       // Clinical report text
\\\

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

This project is licensed under the MIT License.

##  Author

**Yassir**
- Al Akhawayn University - Big Data Analytics
- GPA: 3.6 | Dean's List
- Co-founder: Y-Qubit Solutions & Azer Consulting

##  Acknowledgments

- Dataset: Multiple public fundus image datasets
- ConvNeXt architecture: Facebook AI Research
- Llama 3.2-Vision: Meta AI
- UI Components: shadcn/ui

##  Contact

For questions or collaboration opportunities, please reach out via GitHub issues.

---

** If you find this project helpful, please give it a star!**
