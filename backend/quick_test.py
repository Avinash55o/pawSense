#!/usr/bin/env python3
"""
Quick test script for PawSense backend changes
Tests the new pretrained model integration
"""

import sys
import os

print("=" * 60)
print("🧪 PawSense Backend - Quick Test")
print("=" * 60)
print()

# Test 1: Import the new breed classifier
print("Test 1: Importing breed_classifier module...")
try:
    from models.breed_classifier import get_classifier
    print("✅ PASS: breed_classifier imported successfully")
except Exception as e:
    print(f"❌ FAIL: Could not import breed_classifier")
    print(f"   Error: {e}")
    sys.exit(1)

print()

# Test 2: Import main.py
print("Test 2: Importing main module...")
try:
    import main
    print("✅ PASS: main.py imported successfully")
except Exception as e:
    print(f"❌ FAIL: Could not import main.py")
    print(f"   Error: {e}")
    sys.exit(1)

print()

# Test 3: Check global variables
print("Test 3: Checking global variables...")
try:
    assert hasattr(main, 'breed_classifier'), "breed_classifier not found"
    assert hasattr(main, 'app'), "FastAPI app not found"
    print("✅ PASS: All required globals exist")
except AssertionError as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

print()

# Test 4: Check old functions are removed
print("Test 4: Verifying old functions removed...")
old_functions = ['load_model', 'preprocess_image', 'post_process_results']
removed = []
still_exist = []

for func in old_functions:
    if hasattr(main, func):
        still_exist.append(func)
    else:
        removed.append(func)

if still_exist:
    print(f"⚠️  WARNING: These old functions still exist: {still_exist}")
else:
    print(f"✅ PASS: All old functions removed ({', '.join(removed)})")

print()

# Test 5: Check imports
print("Test 5: Checking imports...")
try:
    # Check that torch is NOT imported in main
    import importlib.util
    spec = importlib.util.spec_from_file_location("main", "main.py")
    
    with open("main.py", "r") as f:
        content = f.read()
        
    has_torch_import = "import torch" in content and "# import torch" not in content
    has_torchvision_import = "import torchvision" in content and "# import torchvision" not in content
    
    if has_torch_import or has_torchvision_import:
        print("⚠️  WARNING: Old torch/torchvision imports still present")
    else:
        print("✅ PASS: Old imports removed")
except Exception as e:
    print(f"⚠️  Could not verify imports: {e}")

print()

# Test 6: Get classifier instance
print("Test 6: Getting classifier instance...")
try:
    classifier = get_classifier()
    print(f"✅ PASS: Classifier instance created: {type(classifier).__name__}")
except Exception as e:
    print(f"❌ FAIL: Could not create classifier")
    print(f"   Error: {e}")
    sys.exit(1)

print()

# Summary
print("=" * 60)
print("📊 Test Summary")
print("=" * 60)
print()
print("✅ All basic tests passed!")
print()
print("Next steps:")
print("1. Run the server: ./test_server.sh")
print("   OR: python3 -m uvicorn main:app --reload")
print()
print("2. Test with curl:")
print("   curl http://localhost:8000/api/health")
print()
print("3. Test prediction with an image")
print()
print("=" * 60)
