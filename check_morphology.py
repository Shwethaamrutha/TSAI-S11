"""
Check if our tokenizer learned common Kannada morphological patterns.
Analyzes suffixes, case markers, and compound word formation.
"""

from tokenizers import Tokenizer
import json


def analyze_morphology():
    """Check what morphological patterns the tokenizer learned."""
    
    print("="*70)
    print("MORPHOLOGICAL PATTERN ANALYSIS")
    print("="*70)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("kannada_tokenizer/tokenizer.json")
    vocab = tokenizer.get_vocab()
    
    print(f"\nTotal vocabulary: {len(vocab)} tokens\n")
    
    # Common Kannada suffixes and case markers
    morphemes = {
        "Case Markers": ["ಗೆ", "ನ್ನು", "ಇಂದ", "ಅಲ್ಲಿ", "ದಲ್ಲಿ"],
        "Verb Suffixes": ["ಅಲು", "ಇಸು", "ತ್ತು", "ತ್ತೇನೆ", "ತ್ತೀರಿ"],
        "Noun Suffixes": ["ತನ", "ಇಕೆ", "ತ್ವ"],
        "Common Endings": ["ವು", "ಯು", "ಆಗಿದೆ", "ಇದೆ", "ಅಲ್ಲ"],
        "Compound Patterns": ["ವಾಗಿ", "ದಂತೆ", "ವಾದ", "ದಲ್ಲಿ"],
    }
    
    print("🔍 CHECKING LEARNED MORPHOLOGICAL PATTERNS:\n")
    
    for category, patterns in morphemes.items():
        print(f"\n{category}:")
        print("-" * 50)
        for pattern in patterns:
            if pattern in vocab:
                token_id = vocab[pattern]
                print(f"  ✅ '{pattern}' → Token ID {token_id}")
            else:
                # Check if it exists as part of longer tokens
                matches = [token for token in vocab.keys() if pattern in token]
                if matches[:3]:  # Show first 3 matches
                    print(f"  ⚠️  '{pattern}' not standalone, but found in:")
                    for match in matches[:3]:
                        print(f"      - '{match}'")
                else:
                    print(f"  ❌ '{pattern}' not learned")
    
    # Test compound word examples from user
    print("\n" + "="*70)
    print("TESTING YOUR COMPOUND WORD EXAMPLES:")
    print("="*70)
    
    examples = [
        ("ಆಗಾಗ", "ಆಗ + ಆಗ"),
        ("ಹೋಗೆಂದ", "ಹೋಗು + ಎಂದ"),
        ("ಚಳಿಗಾಲ", "ಚಳಿ + ಕಾಲ"),
        ("ಕಂಬನಿ", "ಕಣ್ + ಪನಿ"),
        ("ಮಗುವನ್ನು", "ಮಗು + ಅನ್ನು"),
        ("ಪಿತೃವಿಗೆ", "ಪಿತೃ + ಇಗೆ"),
    ]
    
    print("\nHow our tokenizer handles compound words:\n")
    
    for compound, components in examples:
        encoding = tokenizer.encode(compound)
        # Remove special tokens
        tokens = [t for t in encoding.tokens if not (t.startswith('[') and t.endswith(']'))]
        
        print(f"Word: {compound} ({components})")
        print(f"  Tokens: {tokens}")
        print(f"  Count: {len(tokens)} token(s)")
        
        if len(tokens) == 1:
            print(f"  ✅ Learned as single token! (Best case)")
        elif len(tokens) == 2:
            print(f"  ⚠️  Split into 2 parts (Could be better)")
        else:
            print(f"  ❌ Split into {len(tokens)} parts (Over-segmented)")
        print()
    
    # Test case marker attachment
    print("="*70)
    print("TESTING CASE MARKER PATTERNS:")
    print("="*70)
    
    case_examples = [
        "ಮನೆಗೆ",      # house + to
        "ಮನೆಯಿಂದ",    # house + from
        "ಮನೆಯಲ್ಲಿ",   # house + in
        "ಮನೆಯನ್ನು",   # house + object marker
        "ಮನೆಯವರು",   # house + people
    ]
    
    print("\nCase marker attachment patterns:\n")
    
    for word in case_examples:
        encoding = tokenizer.encode(word)
        tokens = [t for t in encoding.tokens if not (t.startswith('[') and t.endswith(']'))]
        print(f"{word:15} → {tokens} ({len(tokens)} tokens)")
    
    # Statistics
    print("\n" + "="*70)
    print("VOCABULARY COMPOSITION:")
    print("="*70)
    
    # Count tokens by length (character count)
    lengths = {}
    for token in vocab.keys():
        # Skip special tokens
        if token.startswith('[') and token.endswith(']'):
            continue
        length = len(token)
        lengths[length] = lengths.get(length, 0) + 1
    
    print("\nToken length distribution:")
    for length in sorted(lengths.keys())[:15]:  # Show first 15
        count = lengths[length]
        bar = "█" * (count // 100)
        print(f"  {length:2} chars: {count:4} tokens {bar}")
    
    # Check for common patterns in vocabulary
    print("\n" + "="*70)
    print("COMMON PATTERNS IN VOCABULARY:")
    print("="*70)
    
    # Find tokens that are likely suffixes (short, common endings)
    potential_suffixes = [
        token for token in vocab.keys() 
        if 1 <= len(token) <= 4 
        and not token.startswith('[')
        and any(char in token for char in "ೆೇೈೊೋೌಂಃ್ು")  # Kannada vowel signs
    ]
    
    print(f"\nPotential suffix tokens (sample of {min(20, len(potential_suffixes))}):")
    for suffix in potential_suffixes[:20]:
        print(f"  '{suffix}'", end="  ")
        if potential_suffixes.index(suffix) % 5 == 4:
            print()
    
    print("\n\n" + "="*70)
    print("INSIGHTS & RECOMMENDATIONS:")
    print("="*70)
    print("""
✅ What BPE Already Does:
   - Learns common suffixes statistically from data
   - Discovers frequent patterns automatically
   - No manual rules needed!

⚠️  Current Limitations:
   - May over-segment rare compound words
   - Depends on training data coverage
   - No explicit morphological knowledge

💡 How to Improve:
   1. More training data → learns more patterns
   2. Morphological pre-processing (advanced)
   3. Increase vocabulary size → capture more compounds
   4. Use morphological analyzer (expert level)

🎯 Your Observation is Correct!
   Kannada morphology (ಗೆ, ಇಂದ, etc.) is similar to English (-ing, -tion).
   BPE learns these automatically from data, but could be enhanced with
   linguistic knowledge for even better tokenization.
""")


if __name__ == "__main__":
    analyze_morphology()

