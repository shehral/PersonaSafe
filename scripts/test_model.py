#!/usr/bin/env python3
"""
Test Gemma 3 Model Loading and Inference
Verifies that Gemma 3 models can be loaded and generate text
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def test_model(model_id, max_new_tokens=50, use_fp16=True):
    """
    Test loading and inference for a Gemma 3 model
    
    Args:
        model_id: HuggingFace model ID (e.g., 'google/gemma-3-4b')
        max_new_tokens: Maximum tokens to generate
        use_fp16: Use float16 for faster inference
    """
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"{'='*60}\n")
    
    # 1. Load tokenizer
    print("📥 Loading tokenizer...")
    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer_time = time.time() - start
        print(f"✅ Tokenizer loaded ({tokenizer_time:.2f}s)")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return False
    
    # 2. Load model
    print("\n📥 Loading model...")
    start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model_time = time.time() - start
        print(f"✅ Model loaded ({model_time:.2f}s)")
        
        # Print model info
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {param_count:,}")
        print(f"   Hidden size: {model.config.hidden_size}")
        print(f"   Layers: {model.config.num_hidden_layers}")
        print(f"   Device: {model.device}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # 3. Test inference
    print("\n🧪 Testing inference...")
    test_prompts = [
        "The future of AI safety is",
        "Fine-tuning language models requires",
        "Google's Gemma 3 models are",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/3 ---")
        print(f"Prompt: {prompt}")
        
        try:
            start = time.time()
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            inference_time = time.time() - start
            
            print(f"Output: {generated_text}")
            print(f"Time: {inference_time:.2f}s")
            print("✅ Inference successful")
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return False
    
    # 4. Memory check
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory:")
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   Allocated: {memory_allocated:.2f}GB")
        print(f"   Reserved: {memory_reserved:.2f}GB")
    
    print(f"\n{'='*60}")
    print(f"✅ ALL TESTS PASSED for {model_id}")
    print(f"{'='*60}\n")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Test Gemma 3 model loading and inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b",
        choices=[
            "google/gemma-3-4b",
            "google/gemma-3-12b",
            "google/gemma-3-27b",
        ],
        help="Gemma 3 model to test"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use float32 instead of float16"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  PersonaSafe - Gemma 3 Model Test")
    print("="*60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠️  CUDA not available, using CPU (slower)")
    
    # Test model
    success = test_model(
        args.model,
        max_new_tokens=args.max_tokens,
        use_fp16=not args.fp32
    )
    
    if success:
        print("🎉 Model test completed successfully!")
        print("\nYou can now use this model for PersonaSafe development.")
        return 0
    else:
        print("❌ Model test failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
