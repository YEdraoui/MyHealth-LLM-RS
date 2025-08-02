# 👁️ Ophthalmology LLM System (BRSET)

An explainable, multi-modal AI assistant for diagnosing and recommending treatment for eye conditions using structured patient data and retinal fundus images.

## 🎯 Project Goal
Build a domain-specialized AI system that combines:
- **Vision Model**: Retinal fundus image classification
- **LLM Integration**: Patient context + clinical reasoning
- **Explainability**: Visual attention + textual rationale

## 📊 Dataset: BRSET
Brazilian Retinal Dataset with:
- 13+ eye conditions
- Patient metadata (age, sex, comorbidities)
- High-quality labeled fundus images

## 🚀 Current Status: Phase 0 ✅
- [x] Project structure setup
- [x] Environment configuration
- [ ] BRSET dataset download
- [ ] Initial EDA analysis

## 📁 Project Structure
```
ophthalmology_llm/
├── data/
│   ├── raw/brset/          # BRSET dataset
│   ├── processed/          # Preprocessed data
│   └── models/            # Trained models
├── vision_model/          # CNN/ResNet classifier
├── llm/                   # Language model components
├── inference/             # Inference pipeline
├── ui/                    # Streamlit dashboard
├── api/                   # FastAPI endpoints
└── notebooks/             # Analysis notebooks
```

## 🔄 Next Steps (Phase 1)
1. Download BRSET dataset
2. Run EDA analysis: `python notebooks/01_brset_eda.py`
3. Build vision classifier
4. Integrate with LLM

## 🏥 Target Use Case
**Clinical Decision Support** for ophthalmologists in:
- Diabetic retinopathy screening
- AMD detection
- Retinal abnormality assessment
- Treatment recommendation
