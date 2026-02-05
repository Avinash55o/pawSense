#!/usr/bin/env python3
"""
Test script for the restructured PawSense backend.
"""
import sys

print("=" * 60)
print("🧪 Testing Restructured PawSense Backend")
print("=" * 60)
print()

# Test 1: Import config
print("Test 1: Importing config...")
try:
    from app.config import get_settings
    settings = get_settings()
    print(f"✅ PASS: Config loaded - {settings.APP_NAME} v{settings.APP_VERSION}")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

print()

# Test 2: Import models
print("Test 2: Importing models...")
try:
    from app.models.breed_classifier import get_classifier
    from app.models.general_qa import get_qa_model
    print("✅ PASS: Models imported successfully")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

print()

# Test 3: Import utils
print("Test 3: Importing utils...")
try:
    from app.utils.breed_info import get_breed_info, get_all_breeds
    breeds = get_all_breeds()
    print(f"✅ PASS: Utils imported - {len(breeds)} breeds available")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

print()

# Test 4: Import services
print("Test 4: Importing services...")
try:
    from app.services.prediction_service import get_prediction_service
    print("✅ PASS: Services imported successfully")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

print()

# Test 5: Import routes
print("Test 5: Importing routes...")
try:
    from app.api.routes import router
    print(f"✅ PASS: Routes imported - {len(router.routes)} routes defined")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

print()

# Test 6: Import main app
print("Test 6: Importing main app...")
try:
    from app.main import app
    print("✅ PASS: Main app imported successfully")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

print()

# Summary
print("=" * 60)
print("📊 Test Summary")
print("=" * 60)
print()
print("✅ All tests passed!")
print()
print("Next steps:")
print("1. Start server: python3 -m uvicorn app.main:app --reload")
print("2. Test health: curl http://localhost:8000/api/system/health")
print("3. View docs: http://localhost:8000/docs")
print()
print("=" * 60)
