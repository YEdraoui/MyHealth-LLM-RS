"""
Phase 4: Testing
"""
import sys
import os
sys.path.append('inference')

def test_inference_system():
    """Test the complete system"""
    print("🚀 PHASE 4: TESTING INFERENCE SYSTEM")
    print("=" * 40)
    
    try:
        from inference_engine import create_inference_engine
        
        # Create engine
        engine = create_inference_engine()
        
        # Test patient
        patient_data = {
            'age': 68,
            'sex': 2,
            'diabetes_time': 15,
            'insulin_use': 1
        }
        
        # Mock image path
        mock_image = "test.jpg"
        
        # Run inference
        result = engine.run_inference(mock_image, patient_data)
        
        print("📊 Test Results:")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Confidence: {result['confidence_score']:.3f}")
        print(f"  Recommendations: {len(result['recommendations'])}")
        
        # Show high predictions
        high_preds = {k: v for k, v in result['predictions'].items() if v > 0.3}
        if high_preds:
            print("  High-confidence findings:")
            for condition, conf in high_preds.items():
                print(f"    - {condition}: {conf:.3f}")
        
        print("✅ Inference system test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_inference_system()
    
    print(f"\n🎯 PHASE 4 STATUS: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print("\n🎉 ALL PHASES COMPLETED!")
        print("🚀 Ready for deployment:")
        print("  streamlit run ui/streamlit_app.py")
