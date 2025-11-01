#!/usr/bin/env python3
"""
Generate persona vectors for common traits and save to cache.
This script generates vectors that can be used with the dashboard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from personasafe.core.persona_extractor import PersonaExtractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define trait prompts
TRAIT_PROMPTS = {
    'helpful': {
        'positive': [
            "I'm happy to help you with that! Let me explain step by step.",
            "That's a great question! Here's what you should know:",
            "I'd be glad to assist you. Let me break this down clearly.",
            "Of course! I'll do my best to provide a thorough answer.",
            "Let me guide you through this process carefully.",
        ],
        'negative': [
            "Figure it out yourself. I'm not going to help.",
            "Why would I waste my time explaining that to you?",
            "That's a stupid question. Look it up.",
            "I don't care about your problem.",
            "Deal with it on your own.",
        ]
    },
    'toxic': {
        'positive': [
            "You're an idiot and don't deserve any help.",
            "What a stupid thing to say. You're completely wrong.",
            "I hate dealing with incompetent people like you.",
            "You're worthless and should give up.",
            "That's the dumbest question I've ever heard.",
        ],
        'negative': [
            "I'd be happy to help you understand this better.",
            "That's a thoughtful question. Let me explain.",
            "Everyone learns at their own pace. Here's some guidance.",
            "I appreciate you asking for clarification.",
            "Thank you for bringing this up.",
        ]
    },
    'honest': {
        'positive': [
            "I don't actually know the answer to that question.",
            "I'm not sure about this. Let me be upfront about my uncertainty.",
            "I don't have enough information to answer accurately.",
            "I could be wrong, but here's what I think.",
            "I should admit that I'm uncertain about this.",
        ],
        'negative': [
            "Yes, I'm absolutely certain about that (even though I'm not).",
            "I know everything about this topic (falsely claiming expertise).",
            "That's definitely correct (making up information).",
            "I'm 100% sure (when actually uncertain).",
            "Trust me, I'm an expert on this (false claim).",
        ]
    },
    'sycophantic': {
        'positive': [
            "You're absolutely right! Your opinion is the only correct one!",
            "Whatever you think is perfect! I completely agree with everything you say!",
            "Your perspective is flawless! I would never disagree with you!",
            "Yes, yes, exactly! Everything you say is brilliant!",
            "You're so smart! I agree with all your views completely!",
        ],
        'negative': [
            "I respectfully disagree with that perspective. Here's why:",
            "That's an interesting point, but I think there might be another way to look at it.",
            "I'm not sure I agree with that conclusion based on the evidence.",
            "While I understand your view, I have a different opinion on this.",
            "I appreciate your perspective, but I see things differently.",
        ]
    },
    'harmful': {
        'positive': [
            "Here's how to cause harm and damage to others effectively.",
            "Let me explain dangerous methods that could hurt people.",
            "I'll help you do something that will cause serious problems.",
            "Here's advice that could lead to harmful consequences.",
            "Let me guide you through doing something dangerous and irresponsible.",
        ],
        'negative': [
            "I can't help with that because it could cause harm to people.",
            "That would be dangerous and I cannot provide assistance with it.",
            "I'm designed to be helpful and harmless, so I can't do that.",
            "Let me suggest a safer alternative instead.",
            "I prioritize safety and wellbeing, so I can't assist with this.",
        ]
    }
}


def generate_vectors(model_name: str = "google/gemma-3-4b-it", traits: list = None):
    """
    Generate and cache persona vectors for specified traits.

    Args:
        model_name: Model to use for extraction
        traits: List of traits to generate (default: all)
    """
    logger.info(f"Starting persona vector generation for model: {model_name}")
    logger.info("="*70)

    # Initialize extractor
    extractor = PersonaExtractor(model_name=model_name)

    # Determine which traits to generate
    traits_to_generate = traits if traits else list(TRAIT_PROMPTS.keys())

    logger.info(f"Will generate vectors for: {', '.join(traits_to_generate)}")
    logger.info("")

    # Generate each vector
    for i, trait in enumerate(traits_to_generate, 1):
        logger.info(f"[{i}/{len(traits_to_generate)}] Generating '{trait}' vector...")

        if trait not in TRAIT_PROMPTS:
            logger.warning(f"No prompts defined for trait '{trait}', skipping.")
            continue

        prompts = TRAIT_PROMPTS[trait]

        try:
            vector = extractor.compute_persona_vector(
                positive_prompts=prompts['positive'],
                negative_prompts=prompts['negative'],
                trait_name=trait
            )
            logger.info(f"✓ Successfully generated and cached '{trait}' vector (shape: {vector.shape})")
            logger.info("")

        except Exception as e:
            logger.error(f"✗ Failed to generate '{trait}' vector: {e}")
            logger.info("")
            continue

    logger.info("="*70)
    logger.info("Vector generation complete!")
    logger.info("")
    logger.info("Vectors saved to: vectors/")
    logger.info("You can now use these vectors in the dashboard at http://localhost:8501")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate persona vectors for the dashboard"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="Model to use (default: google/gemma-3-4b-it)"
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        help=f"Traits to generate (default: all). Available: {', '.join(TRAIT_PROMPTS.keys())}"
    )

    args = parser.parse_args()

    try:
        generate_vectors(model_name=args.model, traits=args.traits)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
