"""
Validate the trained BPE tokenizer for Kannada.
Checks:
1. Vocabulary size > 5000
2. Compression ratio >= 3.2
"""

import os
import json
from tokenizers import Tokenizer
from typing import List, Tuple


def load_tokenizer(tokenizer_dir="kannada_tokenizer"):
    """Load the trained tokenizer."""
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}\n"
            "Please train the tokenizer first: python train_bpe.py"
        )
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer


def calculate_compression_ratio(tokenizer, texts: List[str]) -> Tuple[float, dict]:
    """
    Calculate the compression ratio of the tokenizer.
    
    Compression ratio = (total characters) / (total tokens)
    
    Args:
        tokenizer: Trained tokenizer
        texts: List of text samples
    
    Returns:
        Tuple of (compression_ratio, statistics_dict)
    """
    total_chars = 0
    total_tokens = 0
    
    for text in texts:
        # Count characters (excluding whitespace for more accurate measure)
        chars = len(text.replace(" ", "").replace("\n", ""))
        total_chars += chars
        
        # Encode the text and count tokens
        encoding = tokenizer.encode(text)
        
        # Exclude special tokens ([CLS], [SEP], [PAD], etc.) from token count
        # Special tokens are enclosed in brackets like [CLS], [SEP]
        tokens = encoding.tokens
        actual_tokens = [t for t in tokens if not (t.startswith('[') and t.endswith(']'))]
        tokens_count = len(actual_tokens)
        total_tokens += tokens_count
    
    if total_tokens == 0:
        raise ValueError("No tokens generated. Check your corpus and tokenizer.")
    
    compression_ratio = total_chars / total_tokens
    
    statistics = {
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "compression_ratio": compression_ratio,
        "avg_chars_per_token": compression_ratio,
    }
    
    return compression_ratio, statistics


def validate_tokenizer(
    tokenizer_dir="kannada_tokenizer",
    corpus_file="kannada_corpus.txt",
    num_samples=1000,
    min_vocab_size=5000,
    min_compression_ratio=3.2,
):
    """
    Validate the tokenizer against requirements.
    
    Args:
        tokenizer_dir: Directory containing the trained tokenizer
        corpus_file: Corpus file to test on
        num_samples: Number of samples to use for validation
        min_vocab_size: Minimum required vocabulary size
        min_compression_ratio: Minimum required compression ratio
    
    Returns:
        Dictionary with validation results
    """
    print("="*60)
    print("VALIDATING KANNADA BPE TOKENIZER")
    print("="*60)
    
    # Load tokenizer
    print(f"\n1. Loading tokenizer from {tokenizer_dir}...")
    tokenizer = load_tokenizer(tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()
    print(f"   ✓ Tokenizer loaded successfully")
    print(f"   ✓ Vocabulary size: {vocab_size}")
    
    # Check vocabulary size requirement
    print(f"\n2. Checking vocabulary size requirement (>{min_vocab_size})...")
    vocab_check = vocab_size > min_vocab_size
    if vocab_check:
        print(f"   ✅ PASS: Vocabulary size {vocab_size} > {min_vocab_size}")
    else:
        print(f"   ❌ FAIL: Vocabulary size {vocab_size} <= {min_vocab_size}")
    
    # Load corpus samples for compression ratio test
    print(f"\n3. Loading corpus samples from {corpus_file}...")
    if not os.path.exists(corpus_file):
        print(f"   ⚠️  Corpus file not found: {corpus_file}")
        print(f"   Using sample texts for validation...")
        texts = [
            "ಕನ್ನಡ ದಕ್ಷಿಣ ಭಾರತದ ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆಯಾಗಿದೆ.",
            "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿಯಾಗಿದೆ.",
            "ಕನ್ನಡ ಸಾಹಿತ್ಯವು ಬಹಳ ಪ್ರಾಚೀನವಾಗಿದೆ ಮತ್ತು ಸಮೃದ್ಧವಾಗಿದೆ.",
        ] * 100  # Repeat for better statistics
    else:
        with open(corpus_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            texts = [line.strip() for line in lines[:num_samples] if line.strip()]
    
    print(f"   ✓ Loaded {len(texts)} text samples")
    
    # Calculate compression ratio
    print(f"\n4. Calculating compression ratio (>={min_compression_ratio})...")
    compression_ratio, stats = calculate_compression_ratio(tokenizer, texts)
    
    print(f"   Total characters: {stats['total_characters']:,}")
    print(f"   Total tokens: {stats['total_tokens']:,}")
    print(f"   Compression ratio: {compression_ratio:.4f}")
    
    # Check compression ratio requirement
    compression_check = compression_ratio >= min_compression_ratio
    if compression_check:
        print(f"   ✅ PASS: Compression ratio {compression_ratio:.4f} >= {min_compression_ratio}")
    else:
        print(f"   ❌ FAIL: Compression ratio {compression_ratio:.4f} < {min_compression_ratio}")
    
    # Display some examples
    print(f"\n5. Example tokenizations:")
    example_texts = [
        "ಕನ್ನಡ ಭಾಷೆ",
        "ಬೆಂಗಳೂರು ನಗರ",
        "ಕರ್ನಾಟಕ ರಾಜ್ಯ",
    ]
    
    for text in example_texts:
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        print(f"\n   Text: {text}")
        print(f"   Tokens ({len(tokens)}): {tokens}")
        print(f"   IDs: {encoding.ids}")
    
    # Overall result
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    results = {
        "vocab_size": vocab_size,
        "vocab_size_pass": vocab_check,
        "compression_ratio": compression_ratio,
        "compression_ratio_pass": compression_check,
        "all_checks_pass": vocab_check and compression_check,
        "statistics": stats,
    }
    
    print(f"Vocabulary Size: {vocab_size} (Required: >{min_vocab_size})")
    print(f"  Status: {'✅ PASS' if vocab_check else '❌ FAIL'}")
    print(f"\nCompression Ratio: {compression_ratio:.4f} (Required: >={min_compression_ratio})")
    print(f"  Status: {'✅ PASS' if compression_check else '❌ FAIL'}")
    
    if results["all_checks_pass"]:
        print(f"\n{'='*60}")
        print("🎉 ALL REQUIREMENTS MET! 🎉")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("❌ SOME REQUIREMENTS NOT MET")
        print(f"{'='*60}")
        if not vocab_check:
            print(f"\n💡 Tip: Increase vocab_size in train_bpe.py")
        if not compression_check:
            print(f"\n💡 Tip: Try adjusting min_frequency or using a larger corpus")
    
    # Save results
    results_file = os.path.join(tokenizer_dir, "validation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Kannada BPE tokenizer")
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="kannada_tokenizer",
        help="Directory containing the trained tokenizer"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="kannada_corpus.txt",
        help="Corpus file to test on"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to use for validation"
    )
    
    args = parser.parse_args()
    
    results = validate_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        corpus_file=args.corpus,
        num_samples=args.samples,
    )

