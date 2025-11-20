"""
Setup Verification Script for Credit Risk Assessment System
Run this to check if everything is configured correctly
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_check(passed, message):
    symbol = "✅" if passed else "❌"
    print(f"{symbol} {message}")

def check_python_version():
    print_header("Checking Python Version")
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 8
    print_check(passed, f"Python {version.major}.{version.minor}.{version.micro}")
    if not passed:
        print("   ⚠️  Python 3.8+ recommended")
    return passed

def check_dependencies():
    print_header("Checking Dependencies")
    required = {
        'flask': 'Flask',
        'flask_cors': 'flask-cors',
        'torch': 'torch',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn'
    }
    
    all_good = True
    for module, package in required.items():
        try:
            __import__(module)
            print_check(True, f"{package} installed")
        except ImportError:
            print_check(False, f"{package} NOT installed")
            print(f"   Install with: pip install {package}")
            all_good = False
    
    return all_good

def check_model_file():
    print_header("Checking Model File")
    model_path = Path('credit_risk_pytorch_model_v2.pth')
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print_check(True, f"Model file found ({size_mb:.2f} MB)")
        return True
    else:
        print_check(False, "Model file NOT found")
        print("   Expected: credit_risk_pytorch_model_v2.pth")
        print("   Location: Same directory as app.py")
        print("   Action: Download from Google Colab and place here")
        return False

def check_model_loading():
    print_header("Testing Model Loading")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        # Try loading the model
        checkpoint = torch.load('credit_risk_pytorch_model_v2.pth', 
                               map_location=torch.device('cpu'))
        
        required_keys = ['model_state_dict', 'scaler', 'feature_columns', 'label_encoders']
        all_keys = True
        
        for key in required_keys:
            if key in checkpoint:
                print_check(True, f"'{key}' found in model")
            else:
                print_check(False, f"'{key}' missing from model")
                all_keys = False
        
        if all_keys:
            print(f"\n   📊 Features: {len(checkpoint['feature_columns'])}")
            print(f"   🏷️  Encoders: {len(checkpoint['label_encoders'])}")
            print(f"   📉 Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        return all_keys
    except FileNotFoundError:
        print_check(False, "Model file not found")
        return False
    except Exception as e:
        print_check(False, f"Error loading model: {str(e)}")
        return False

def check_port_availability():
    print_header("Checking Port Availability")
    import socket
    
    port = 5000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    
    if result == 0:
        print_check(False, f"Port {port} is already in use")
        print("   Action: Stop other services or change port in app.py")
        return False
    else:
        print_check(True, f"Port {port} is available")
        return True

def test_api_imports():
    print_header("Testing API Imports")
    try:
        from app import ImprovedCreditRiskNN, load_model
        print_check(True, "API imports successful")
        return True
    except ImportError as e:
        print_check(False, f"Import error: {str(e)}")
        print("   Make sure app.py is in the current directory")
        return False
    except Exception as e:
        print_check(False, f"Error: {str(e)}")
        return False

def check_folder_structure():
    print_header("Checking Folder Structure")
    
    files_to_check = {
        'app.py': 'Backend API file',
        'requirements.txt': 'Dependencies file',
        'credit_risk_pytorch_model_v2.pth': 'Trained model'
    }
    
    all_good = True
    for filename, description in files_to_check.items():
        if Path(filename).exists():
            print_check(True, f"{filename} ({description})")
        else:
            print_check(False, f"{filename} missing ({description})")
            all_good = False
    
    return all_good

def run_full_test():
    print("\n" + "🔍 CREDIT RISK ASSESSMENT SYSTEM - SETUP VERIFICATION")
    
    checks = [
        ("Python Version", check_python_version),
        ("Folder Structure", check_folder_structure),
        ("Dependencies", check_dependencies),
        ("Model File", check_model_file),
        ("Model Loading", check_model_loading),
        ("Port Availability", check_port_availability)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")
            results[name] = False
    
    # Summary
    print_header("SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n   Checks Passed: {passed}/{total}")
    
    if passed == total:
        print("\n   ✅ All checks passed! You're ready to run the API.")
        print("\n   Next steps:")
        print("   1. Run: python app.py")
        print("   2. Open frontend/index.html in browser")
        print("   3. Test the prediction system")
    else:
        print("\n   ⚠️  Some checks failed. Please fix the issues above.")
        print("\n   Common fixes:")
        print("   • Install dependencies: pip install -r requirements.txt")
        print("   • Download model from Colab")
        print("   • Ensure correct folder structure")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    run_full_test()