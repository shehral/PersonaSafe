#!/usr/bin/env python3
"""
Quick Demo: Persona Vector Extraction
Demonstrates basic persona vector extraction from Gemma 3
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def extract_hidden_states(model, tokenizer, text):
    """Extract hidden states from model for given text"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get middle layer hidden states
    hidden_states = outputs.hidden_states
    middle_layer = len(hidden_states) // 2
    
    # Average over sequence length
    hidden_state = hidden_states[middle_layer].mean(dim=1).squeeze()
    
    return hidden_state.cpu().numpy()

def compute_persona_vector(model, tokenizer, trait, examples=None):
    """
    Compute persona vector for a trait using contrastive prompting
    
    Args:
        model: Gemma 3 model
        tokenizer: Tokenizer
        trait: Trait name (e.g., 'helpful', 'toxic')
        examples: Optional custom examples
    """
    
    # Default examples for common traits
    default_examples = {
        'helpful': {
            'positive': [
                "I'm happy to help you with that! Let me explain step by step.",
                "That's a great question! Here's what you should know:",
                "I'd be glad to assist you. Let me break this down clearly.",
            ],
            'negative': [
                "Figure it out yourself. I'm not going to help.",
                "Why would I waste my time explaining that to you?",
                "That's a stupid question. Look it up.",
            ]
        },
        'toxic': {
            'positive': [
                "You're an idiot and don't deserve any help.",
                "What a stupid thing to say. You're completely wrong.",
                "I hate dealing with incompetent people like you.",
            ],
            'negative': [
                "I'd be happy to help you understand this better.",
                "That's a thoughtful question. Let me explain.",
                "Everyone learns at their own pace. Here's some guidance.",
            ]
        },
        'honest': {
            'positive': [
                "I don't actually know the answer to that question.",
                "I'm not sure about this. Let me be upfront about my uncertainty.",
                "I don't have enough information to answer accurately.",
            ],
            'negative': [
                "Yes, I'm absolutely certain about that (even though I'm not).",
                "I know everything about this topic (falsely claiming expertise).",
                "That's definitely correct (making up information).",
            ]
        },
    }
    
    examples = examples or default_examples.get(trait, default_examples['helpful'])
    
    print(f"\n📊 Computing persona vector for: '{trait}'")
    print(f"   Using {len(examples['positive'])} positive + {len(examples['negative'])} negative examples")
    
    # Extract hidden states for positive examples
    positive_states = []
    for text in examples['positive']:
        state = extract_hidden_states(model, tokenizer, text)
        positive_states.append(state)
    positive_mean = np.mean(positive_states, axis=0)
    
    # Extract hidden states for negative examples
    negative_states = []
    for text in examples['negative']:
        state = extract_hidden_states(model, tokenizer, text)
        negative_states.append(state)
    negative_mean = np.mean(negative_states, axis=0)
    
    # Compute persona vector (contrastive)
    persona_vector = positive_mean - negative_mean
    
    # Normalize
    persona_vector = persona_vector / np.linalg.norm(persona_vector)
    
    print(f"   ✅ Persona vector shape: {persona_vector.shape}")
    print(f"   ✅ Persona vector norm: {np.linalg.norm(persona_vector):.4f}")
    
    return persona_vector

def score_text(model, tokenizer, persona_vector, text):
    """Score text against persona vector"""
    hidden_state = extract_hidden_states(model, tokenizer, text)
    score = np.dot(hidden_state, persona_vector)
    return score

def demo_persona_extraction(model_id, trait='helpful'):
    """Run persona extraction demo"""
    
    print(f"\n{'='*70}")
    print(f"  PersonaSafe Demo: Persona Vector Extraction")
    print(f"  Model: {model_id}")
    print(f"  Trait: {trait}")
    print(f"{'='*70}\n")
    
    # Load model and tokenizer
    print("📥 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print(f"✅ Model loaded on {model.device}\n")
    
    # Extract persona vector
    persona_vector = compute_persona_vector(model, tokenizer, trait)
    
    # Test on example texts
    print(f"\n🧪 Testing persona detection:\n")
    
    test_texts = {
        'helpful': [
            "I'd be happy to explain that concept to you!",
            "Figure it out yourself, I'm busy.",
            "Let me break this down step by step for you.",
            "Why are you bothering me with this?",
        ],
        'toxic': [
            "You're completely incompetent and stupid.",
            "I appreciate your perspective on this.",
            "That's the dumbest thing I've ever heard.",
            "Thank you for sharing your thoughts.",
        ],
        'honest': [
            "I don't know the answer to that.",
            "I'm 100% certain this is correct (when uncertain).",
            "I'm not sure about this, let me clarify.",
            "I know everything about this topic (falsely).",
        ],
    }
    
    texts = test_texts.get(trait, test_texts['helpful'])
    
    for i, text in enumerate(texts, 1):
        score = score_text(model, tokenizer, persona_vector, text)
        
        # Determine if high or low
        symbol = "🔴" if score > 0 else "🟢"
        direction = "HIGH" if score > 0 else "LOW"
        
        print(f"{symbol} [{direction:>4}] Score: {score:+.3f}")
        print(f"   Text: \"{text}\"")
        print()
    
    print(f"{'='*70}")
    print(f"✅ Demo complete!")
    print(f"\nInterpretation:")
    print(f"  • Positive scores → Text exhibits '{trait}' trait")
    print(f"  • Negative scores → Text lacks '{trait}' trait")
    print(f"  • Higher magnitude → Stronger signal")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Demo: Extract and test persona vectors from Gemma 3"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b",
        help="Gemma 3 model to use"
    )
    parser.add_argument(
        "--trait",
        type=str,
        default="helpful",
        choices=['helpful', 'toxic', 'honest'],
        help="Trait to extract"
    )
    
    args = parser.parse_args()
    
    try:
        demo_persona_extraction(args.model, args.trait)
        return 0
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
