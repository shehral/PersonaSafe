#!/usr/bin/env python3
"""
Download Gemma 3 Models for PersonaSafe
Handles downloading and caching of Gemma 3 models from HuggingFace
"""

import argparse
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login, HfApi
from tqdm import tqdm

# Gemma 3 model information
GEMMA3_MODELS = {
    '4b': {
        'id': 'google/gemma-3-4b',
        'size': '~6GB',
        'params': '2.8B',
        'description': 'Development model - fast iteration, multimodal',
        'recommended': True,
    },
    '12b': {
        'id': 'google/gemma-3-12b',
        'size': '~25GB',
        'params': '11B',
        'description': 'Production model - impressive demos, multimodal',
        'recommended': True,
    },
    '27b': {
        'id': 'google/gemma-3-27b',
        'size': '~55GB',
        'params': '24B',
        'description': 'Flagship model - best performance, optional',
        'recommended': False,
    },
}

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def check_authentication():
    """Check if user is logged in to HuggingFace"""
    print_header("Authentication Check")
    
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    
    if token_path.exists():
        print("✅ HuggingFace token found")
        return True
    else:
        print("❌ Not logged in to HuggingFace")
        print("\nPlease run: huggingface-cli login")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return False

def check_model_access(model_id):
    """Check if user has access to the model"""
    try:
        api = HfApi()
        model_info = api.model_info(model_id)
        
        if model_info.gated:
            print(f"⚠️  {model_id} is gated - you need to accept the license")
            print(f"   Visit: https://huggingface.co/{model_id}")
            return False
        
        return True
    except Exception as e:
        print(f"⚠️  Could not check access for {model_id}: {e}")
        return False

def download_model(model_id, model_name, allow_patterns=None):
    """Download a Gemma 3 model"""
    
    print_header(f"Downloading {model_name}")
    
    print(f"Model ID: {model_id}")
    print(f"Size: {GEMMA3_MODELS[model_name]['size']}")
    print(f"Parameters: {GEMMA3_MODELS[model_name]['params']}")
    print(f"Description: {GEMMA3_MODELS[model_name]['description']}")
    
    # Check access
    if not check_model_access(model_id):
        print(f"\n❌ Cannot download {model_id}")
        print("Please accept the license on HuggingFace and try again.")
        return False
    
    try:
        print("\n📥 Downloading (this may take a while)...\n")
        
        cache_dir = snapshot_download(
            repo_id=model_id,
            allow_patterns=allow_patterns,
            resume_download=True,
        )
        
        print(f"\n✅ Downloaded to: {cache_dir}")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def list_models():
    """List available Gemma 3 models"""
    print_header("Available Gemma 3 Models")
    
    for name, info in GEMMA3_MODELS.items():
        marker = "⭐" if info['recommended'] else "  "
        print(f"{marker} {name.upper()}: {info['id']}")
        print(f"   Size: {info['size']} | Params: {info['params']}")
        print(f"   {info['description']}")
        if info['recommended']:
            print(f"   ✅ Recommended for PersonaSafe")
        print()

def download_recommended(assume_yes: bool = False):
    """Download recommended models for PersonaSafe"""
    print_header("Downloading Recommended Models")
    
    print("This will download:")
    print("1. Gemma 3 4B (~6GB) - For development")
    print("2. Gemma 3 12B (~25GB) - For demos")
    print("\nTotal: ~31GB\n")
    
    if not assume_yes:
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            print("Download cancelled.")
            return False
    
    results = {}
    
    # Download 4B
    results['4b'] = download_model(
        GEMMA3_MODELS['4b']['id'],
        '4b'
    )
    
    # Download 12B
    results['12b'] = download_model(
        GEMMA3_MODELS['12b']['id'],
        '12b'
    )
    
    # Summary
    print_header("Download Summary")
    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} Gemma 3 {model.upper()}: {'Success' if success else 'Failed'}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n🎉 All recommended models downloaded successfully!")
        print("\nNext steps:")
        print("1. Run: python scripts/verify_setup.py")
        print("2. Run: python scripts/test_model.py --model google/gemma-3-4b")
        print("3. Start developing PersonaSafe!")
    else:
        print("\n⚠️  Some downloads failed. Please check the errors above.")
    
    return all_success

def main():
    parser = argparse.ArgumentParser(
        description="Download Gemma 3 models for PersonaSafe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['4b', '12b', '27b', 'all'],
        help="Specific model to download"
    )
    parser.add_argument(
        "--recommended",
        action="store_true",
        help="Download recommended models (4B + 12B)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Assume yes for all prompts (non-interactive)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  PersonaSafe - Gemma 3 Model Downloader")
    print("="*70)
    
    # List models
    if args.list:
        list_models()
        return 0
    
    # Check authentication
    if not check_authentication():
        return 1
    
    # Download recommended
    if args.recommended or (not args.model):
        success = download_recommended(assume_yes=args.yes)
        return 0 if success else 1
    
    # Download specific model
    if args.model == 'all':
        results = {}
        for name in GEMMA3_MODELS:
            results[name] = download_model(
                GEMMA3_MODELS[name]['id'],
                name
            )
        
        success = all(results.values())
        return 0 if success else 1
    
    else:
        success = download_model(
            GEMMA3_MODELS[args.model]['id'],
            args.model
        )
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
