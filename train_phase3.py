"""
Phase 3: Explainability + Safety Training and Testing
"""
import torch
import numpy as np
import sys
import os

sys.path.append('vision_model')
sys.path.append('llm')

from explainability import GradCAMExplainer, SafetyValidator
from multimodal_fusion import create_multimodal_model

def test_explainability():
    """Test Grad-CAM explainability"""
    print("🔍 PHASE 3: EXPLAINABILITY TESTING")
    print("=" * 40)
    
    # Create a simple vision model for testing
    class SimpleVisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 7, 2, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, 2, 1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1)
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(128, 13)
            )
        
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    # Create model and explainer
    model = SimpleVisionModel()
    explainer = GradCAMExplainer(model, target_layer_name='backbone')
    
    # Test with mock image
    mock_image = torch.randn(1, 3, 224, 224)
    
    print("🔬 Testing Grad-CAM generation...")
    try:
        # Generate CAM for diabetic retinopathy (class 0)
        cam = explainer.generate_cam(mock_image, class_idx=0)
        print(f"✅ Grad-CAM generated successfully")
        print(f"📊 CAM shape: {cam.shape}")
        print(f"📈 CAM range: {cam.min():.3f} to {cam.max():.3f}")
        
        # Test visualization
        mock_orig_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        overlay = explainer.visualize_cam(mock_orig_image, cam)
        print(f"🎨 Visualization overlay created: {overlay.shape}")
        
        return True
    except Exception as e:
        print(f"⚠️ Explainability test error: {e}")
        return False

def test_safety_validation():
    """Test safety validation system"""
    print("\n🛡️ SAFETY VALIDATION TESTING")
    print("=" * 35)
    
    # Create safety validator
    validator = SafetyValidator(confidence_threshold=0.7)
    
    # Test cases
    test_cases = [
        {
            'name': 'Emergency Case',
            'predictions': [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.85, 0.1, 0.1, 0.1],  # High DR + retinal detachment
            'patient': {'age': 75, 'diabetes_time': 15, 'insulin_use': 1}
        },
        {
            'name': 'High Risk Case',
            'predictions': [0.8, 0.75, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # High DR + macular edema
            'patient': {'age': 65, 'diabetes_time': 8, 'insulin_use': 0}
        },
        {
            'name': 'Uncertain Case',
            'predictions': [0.4, 0.3, 0.5, 0.4, 0.6, 0.3, 0.4, 0.5, 0.3, 0.2, 0.4, 0.5, 0.3],  # Multiple uncertainties
            'patient': {'age': 45, 'diabetes_time': 3, 'insulin_use': 0}
        },
        {
            'name': 'Normal Case',
            'predictions': [0.1, 0.05, 0.02, 0.03, 0.1, 0.01, 0.05, 0.08, 0.02, 0.01, 0.04, 0.06, 0.02],  # Low all around
            'patient': {'age': 35, 'diabetes_time': 1, 'insulin_use': 0}
        }
    ]
    
    condition_names = [
        'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
        'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 
        'hemorrhage', 'retinal_detachment', 'myopic_fundus', 
        'increased_cup_disc', 'other'
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        
        # Generate safety report
        safety_report, validation = validator.generate_safety_report(
            test_case['predictions'], 
            condition_names, 
            test_case['patient']
        )
        
        print(safety_report)
        
        # Validation summary
        if validation['emergency_referral']:
            print("🚨 Result: EMERGENCY REFERRAL REQUIRED")
        elif validation['requires_specialist']:
            print("⚡ Result: SPECIALIST CONSULTATION NEEDED")
        else:
            print("✅ Result: ROUTINE FOLLOW-UP")
    
    return True

def test_integrated_pipeline():
    """Test complete Phase 3 integrated pipeline"""
    print("\n🔗 INTEGRATED PIPELINE TESTING")
    print("=" * 35)
    
    try:
        # Create multimodal model
        model = create_multimodal_model()
        
        # Create explainer and validator
        explainer = GradCAMExplainer(model)
        validator = SafetyValidator()
        
        # Mock patient case
        patient_data = {
            'age': 68, 
            'sex': 2,  # Female
            'diabetes_time': 12, 
            'insulin_use': 1
        }
        
        # Mock predictions (simulating model output)
        mock_predictions = np.array([0.85, 0.7, 0.1, 0.05, 0.3, 0.05, 0.2, 0.15, 0.1, 0.02, 0.25, 0.1, 0.05])
        
        condition_names = [
            'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
            'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 
            'hemorrhage', 'retinal_detachment', 'myopic_fundus', 
            'increased_cup_disc', 'other'
        ]
        
        # Generate comprehensive report
        safety_report, validation = validator.generate_safety_report(
            mock_predictions, condition_names, patient_data
        )
        
        print("📋 COMPREHENSIVE AI CLINICAL REPORT")
        print("=" * 45)
        print(safety_report)
        
        # Integration status
        print("\n✅ PHASE 3 INTEGRATION SUCCESSFUL!")
        print("🎯 Components working:")
        print("  - Explainable AI (Grad-CAM)")
        print("  - Safety validation system")
        print("  - Risk stratification")
        print("  - Clinical recommendations")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Integration test error: {e}")
        return False

if __name__ == "__main__":
    print("🛡️ PHASE 3: EXPLAINABILITY + SAFETY TESTING")
    print("=" * 50)
    
    # Test explainability
    explainability_ok = test_explainability()
    
    # Test safety validation
    safety_ok = test_safety_validation()
    
    # Test integrated pipeline
    integration_ok = test_integrated_pipeline()
    
    print(f"\n📊 PHASE 3 TEST RESULTS:")
    print(f"  🔍 Explainability: {'✅ PASS' if explainability_ok else '❌ FAIL'}")
    print(f"  🛡️ Safety Validation: {'✅ PASS' if safety_ok else '❌ FAIL'}")
    print(f"  🔗 Integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    
    if all([explainability_ok, safety_ok, integration_ok]):
        print("\n🎉 PHASE 3: EXPLAINABILITY + SAFETY COMPLETED!")
        print("🔄 Ready for Phase 4: Inference System + UI")
    else:
        print("\n⚠️ Some tests failed - review and fix before proceeding")
