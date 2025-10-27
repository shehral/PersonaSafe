#!/usr/bin/env python3
"""
PersonaSafe Setup Verification Script
Tests that all dependencies and Gemma 3 models are properly installed
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def print_status(check, status, message=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {check}")
    if message:
        print(f"   → {message}")

def check_python_version():
    """Check Python version is 3.10+"""
    print_header("Python Version")
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 10
    print_status(
        "Python 3.10+",
        is_valid,
        f"Found Python {version.major}.{version.minor}.{version.micro}"
    )
    return is_valid

def check_dependencies():
    """Check required Python packages"""
    print_header("Dependencies")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'tqdm': 'Progress bars',
        'matplotlib': 'Plotting',
    }
    
    all_installed = True
    for package, name in required.items():
        try:
            __import__(package)
            print_status(name, True, f"{package} installed")
        except ImportError:
            print_status(name, False, f"{package} NOT installed")
            all_installed = False
    
    return all_installed

def check_cuda():
    """Check CUDA availability"""
    print_header("GPU/CUDA")
    
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_status(
                "CUDA Available",
                True,
                f"{gpu_name} ({gpu_memory:.1f}GB)"
            )
        else:
            print_status(
                "CUDA Available",
                False,
                "Will use CPU (slower but works)"
            )
        
        return True  # Not required, just informational
    except Exception as e:
        print_status("CUDA Check", False, str(e))
        return True

def check_huggingface_token():
    """Check HuggingFace authentication"""
    print_header("HuggingFace Authentication")
    
    from pathlib import Path
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    
    if token_path.exists():
        print_status(
            "HuggingFace Token",
            True,
            "Logged in (required for Gemma 3 models)"
        )
        return True
    else:
        print_status(
            "HuggingFace Token",
            False,
            "Run 'huggingface-cli login' to authenticate"
        )
        return False

def check_gemma_models():
    """Check if Gemma 3 models are accessible"""
    print_header("Gemma 3 Models")
    
    try:
        from transformers import AutoTokenizer
        
        models = {
            'google/gemma-3-4b': 'Gemma 3 4B (Development)',
            'google/gemma-3-12b': 'Gemma 3 12B (Production)',
            'google/gemma-3-27b': 'Gemma 3 27B (Flagship - Optional)',
        }
        
        any_available = False
        for model_id, name in models.items():
            try:
                # Just try to load tokenizer (fast check)
                AutoTokenizer.from_pretrained(model_id)
                print_status(name, True, f"{model_id} accessible")
                any_available = True
            except Exception as e:
                error_msg = str(e)
                if "gated" in error_msg.lower() or "access" in error_msg.lower():
                    print_status(name, False, "Need to accept license on HuggingFace")
                else:
                    print_status(name, False, "Not downloaded yet")
        
        return any_available
    except Exception as e:
        print_status("Model Check", False, str(e))
        return False

def check_project_structure():
    """Check project directory structure"""
    print_header("Project Structure")
    
    required_dirs = [
        'personasafe',
        'tests',
        'data',
        'vectors',
        'scripts',
    ]
    
    all_exist = True
    for dirname in required_dirs:
        exists = os.path.isdir(dirname)
        if not exists:
            # Create it
            try:
                os.makedirs(dirname, exist_ok=True)
                print_status(f"{dirname}/", True, "Created")
            except:
                print_status(f"{dirname}/", False, "Could not create")
                all_exist = False
        else:
            print_status(f"{dirname}/", True, "Exists")
    
    return all_exist

def check_env_file():
    """Check .env configuration"""
    print_header("Configuration")
    
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print_status(".env file", True, "Configuration file exists")
        return True
    elif env_example_path.exists():
        print_status(
            ".env file",
            False,
            "Copy .env.example to .env and configure"
        )
        return False
    else:
        print_status(".env file", False, "No configuration found")
        return False

def print_summary(results):
    """Print final summary"""
    print_header("Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Checks passed: {passed}/{total}\n")
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED! You're ready to start developing!")
        print("\nNext steps:")
        print("1. Run: source venv/bin/activate")
        print("2. Try: python -c 'from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained(\"google/gemma-3-4b\"))'")
        print("3. Start building PersonaSafe!")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
        print("\nCommon fixes:")
        
        if not results.get('huggingface'):
            print("- Run: huggingface-cli login")
            print("  Get token from: https://huggingface.co/settings/tokens")
        
        if not results.get('models'):
            print("- Accept Gemma 3 license: https://huggingface.co/google/gemma-3-4b")
            print("- Download models: huggingface-cli download google/gemma-3-4b")
        
        if not results.get('dependencies'):
            print("- Install missing packages: pip install -r requirements.txt")
        
        return 1

def main():
    print("\n" + "="*60)
    print("  PersonaSafe Setup Verification")
    print("  Checking Gemma 3 Installation")
    print("="*60)
    
    results = {
        'python': check_python_version(),
        'dependencies': check_dependencies(),
        'cuda': check_cuda(),
        'huggingface': check_huggingface_token(),
        'models': check_gemma_models(),
        'structure': check_project_structure(),
        'config': check_env_file(),
    }
    
    return print_summary(results)

if __name__ == "__main__":
    sys.exit(main())
