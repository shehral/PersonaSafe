# PersonaSafe Scripts

This directory contains utility scripts for PersonaSafe development with Gemma 3 models.

## 📜 Available Scripts

### 1. `verify_setup.py` ✅
**Purpose**: Verify your development environment is properly configured

**Usage**:
```bash
python scripts/verify_setup.py
```

**Checks**:
- ✅ Python version (3.10+)
- ✅ Required dependencies installed
- ✅ CUDA/GPU availability
- ✅ HuggingFace authentication
- ✅ Gemma 3 model access
- ✅ Project directory structure
- ✅ Configuration files

**When to use**: After running `setup.sh` and before starting development

---

### 2. `download_models.py` 📥
**Purpose**: Download Gemma 3 models from HuggingFace

**Usage**:
```bash
# Download recommended models (4B + 12B)
python scripts/download_models.py --recommended

# Download specific model
python scripts/download_models.py --model 4b
python scripts/download_models.py --model 12b
python scripts/download_models.py --model 27b

# Download all models
python scripts/download_models.py --model all

# List available models
python scripts/download_models.py --list
```

**Models**:
- **4B** (2.8B params, ~6GB): Development, fast iteration, multimodal ⭐
- **12B** (11B params, ~25GB): Production demos, best for presentation ⭐
- **27B** (24B params, ~55GB): Flagship performance, optional

**When to use**: After HuggingFace login, before development

---

### 3. `test_model.py` 🧪
**Purpose**: Test that Gemma 3 models load and generate text correctly

**Usage**:
```bash
# Test Gemma 3 4B (default)
python scripts/test_model.py

# Test specific model
python scripts/test_model.py --model google/gemma-3-4b
python scripts/test_model.py --model google/gemma-3-12b
python scripts/test_model.py --model google/gemma-3-27b

# Use float32 (slower but more precise)
python scripts/test_model.py --fp32

# Generate more tokens
python scripts/test_model.py --max-tokens 100
```

**Tests**:
- ✅ Tokenizer loading
- ✅ Model loading
- ✅ Text generation (3 test prompts)
- ✅ GPU memory usage
- ✅ Inference speed

**When to use**: After downloading models, before starting development

---

### 4. `quick_demo.py` 🎬
**Purpose**: Quick demonstration of persona vector extraction

**Usage**:
```bash
# Demo with 'helpful' trait (default)
python scripts/quick_demo.py

# Demo with specific trait
python scripts/quick_demo.py --trait helpful
python scripts/quick_demo.py --trait toxic
python scripts/quick_demo.py --trait honest

# Use different model
python scripts/quick_demo.py --model google/gemma-3-12b --trait helpful
```

**What it does**:
1. Loads Gemma 3 model
2. Extracts persona vector for specified trait
3. Tests vector on example texts
4. Shows scores (positive = trait present, negative = trait absent)

**When to use**: To understand how persona vectors work, or for quick testing

---

## 🚀 Recommended Workflow

### First Time Setup:
```bash
# 1. Run setup script
./setup.sh

# 2. Verify installation
python scripts/verify_setup.py

# 3. Download models (if not done in setup)
python scripts/download_models.py --recommended

# 4. Test a model
python scripts/test_model.py --model google/gemma-3-4b

# 5. Try quick demo
python scripts/quick_demo.py --trait helpful
```

### During Development:
```bash
# Test model before using
python scripts/test_model.py --model google/gemma-3-4b

# Quick persona extraction test
python scripts/quick_demo.py --trait toxic

# Verify environment after changes
python scripts/verify_setup.py
```

---

## 📊 Expected Output Examples

### verify_setup.py ✅
```
==============================================================
  PersonaSafe Setup Verification
  Checking Gemma 3 Installation
==============================================================

==============================================================
  Python Version
==============================================================

✅ Python 3.10+
   → Found Python 3.11.5

...

==============================================================
  Summary
==============================================================

Checks passed: 7/7

🎉 ALL CHECKS PASSED! You're ready to start developing!
```

### test_model.py ✅
```
============================================================
Testing: google/gemma-3-4b
============================================================

📥 Loading tokenizer...
✅ Tokenizer loaded (0.32s)

📥 Loading model...
✅ Model loaded (5.12s)
   Parameters: 2,823,456,768
   Hidden size: 2048
   Layers: 26
   Device: cuda:0

🧪 Testing inference...

--- Test 1/3 ---
Prompt: The future of AI safety is
Output: The future of AI safety is critical for ensuring...
Time: 1.23s
✅ Inference successful

...

✅ ALL TESTS PASSED for google/gemma-3-4b
```

### quick_demo.py ✅
```
======================================================================
  PersonaSafe Demo: Persona Vector Extraction
  Model: google/gemma-3-4b
  Trait: helpful
======================================================================

📊 Computing persona vector for: 'helpful'
   Using 3 positive + 3 negative examples
   ✅ Persona vector shape: (2048,)
   ✅ Persona vector norm: 1.0000

🧪 Testing persona detection:

🔴 [HIGH] Score: +0.187
   Text: "I'd be happy to explain that concept to you!"

🟢 [ LOW] Score: -0.234
   Text: "Figure it out yourself, I'm busy."

...

✅ Demo complete!
```

---

## 🐛 Troubleshooting

### "HuggingFace token not found"
```bash
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens
```

### "Model is gated - need to accept license"
Visit: https://huggingface.co/google/gemma-3-4b  
Click "Agree and access repository"

### "CUDA out of memory"
Try:
```bash
# Use smaller model
python scripts/test_model.py --model google/gemma-3-4b

# Or use CPU (slower)
CUDA_VISIBLE_DEVICES="" python scripts/test_model.py
```

### "Import Error: No module named X"
```bash
pip install -r requirements.txt
```

---

## 💡 Tips

### Speed up downloads:
Set HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
python scripts/download_models.py --recommended
```

### Save GPU memory:
Use float16 (default) instead of float32:
```bash
python scripts/test_model.py  # Uses fp16 by default
```

### Quick model check:
```bash
python scripts/test_model.py --max-tokens 20  # Faster, shorter outputs
```

---

## 📝 Notes

- All scripts assume you're in the project root directory
- Scripts use Gemma 3 models by default (not Gemma 2)
- GPU is recommended but not required (CPU works, just slower)
- Scripts are designed to be self-contained and easy to understand
- Check script source code for more options and customization

---

**Questions?** Check the main documentation or run scripts with `--help`:
```bash
python scripts/verify_setup.py --help
python scripts/test_model.py --help
python scripts/download_models.py --help
python scripts/quick_demo.py --help
```
