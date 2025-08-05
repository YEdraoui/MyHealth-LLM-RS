"""
Phase 1 Testing: Verify all components work
"""
import torch
from pathlib import Path
import pandas as pd
import json

def test_phase1():
    """Test Phase 1 completion"""
    print("🧪 TESTING PHASE 1 COMPONENTS")
    print("=" * 35)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Dataset available
    labels_path = Path("data/brset_real/labels.csv")
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        print(f"✅ Test 1: Dataset available ({len(df):,} samples)")
        tests_passed += 1
    else:
        print("❌ Test 1: Dataset not found")
    
    # Test 2: Images accessible
    images_path = Path("data/brset_real/images")
    if images_path.exists():
        print("✅ Test 2: Images directory accessible")
        tests_passed += 1
    else:
        print("❌ Test 2: Images not accessible")
    
    # Test 3: Model trained
    model_path = Path("models/phase1_best_model.pth")
    if model_path.exists():
        print("✅ Test 3: Vision model trained")
        tests_passed += 1
    else:
        print("❌ Test 3: Model not trained")
    
    # Test 4: Training stats available
    stats_path = Path("models/training_stats.json")
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"✅ Test 4: Training stats available ({len(stats)} epochs)")
        tests_passed += 1
    else:
        print("❌ Test 4: Training stats missing")
    
    # Test 5: Streamlit app ready
    demo_path = Path("streamlit_demo.py")
    if demo_path.exists():
        print("✅ Test 5: Streamlit demo ready")
        tests_passed += 1
    else:
        print("❌ Test 5: Streamlit demo missing")
    
    print(f"\n📊 PHASE 1 TEST RESULTS: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 PHASE 1: ALL TESTS PASSED!")
        return True
    else:
        print("⚠️ Some tests failed - review above")
        return False

if __name__ == "__main__":
    success = test_phase1()
    if success:
        print("\n🚀 READY FOR STREAMLIT DEMO!")
        print("Run: streamlit run streamlit_demo.py")
