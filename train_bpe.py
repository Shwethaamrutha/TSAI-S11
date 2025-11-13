"""
Train a BPE tokenizer for Kannada language.
The tokenizer will have >5000 tokens and achieve compression ratio >3.2
"""

import os
import json
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
from tokenizers.processors import TemplateProcessing


def train_kannada_bpe(
    corpus_file="kannada_corpus.txt",
    vocab_size=8000,  # Target vocabulary size (>5000 required)
    output_dir="kannada_tokenizer",
    min_frequency=2,
):
    """
    Train a BPE tokenizer for Kannada.
    
    Args:
        corpus_file: Path to the Kannada text corpus
        vocab_size: Target vocabulary size (must be >5000)
        output_dir: Directory to save the trained tokenizer
        min_frequency: Minimum frequency for a token to be included
    
    Returns:
        Trained tokenizer
    """
    
    if vocab_size <= 5000:
        print(f"⚠️  Warning: vocab_size={vocab_size} must be >5000")
        vocab_size = 5001
    
    print(f"Training BPE tokenizer for Kannada...")
    print(f"  Corpus: {corpus_file}")
    print(f"  Target vocab size: {vocab_size}")
    print(f"  Min frequency: {min_frequency}")
    
    # Check if corpus exists
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_file}\n"
            "Please run: python prepare_corpus.py"
        )
    
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Set normalizer (for text preprocessing)
    # For Kannada, we use NFC normalization (combines Unicode characters properly)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),  # Unicode normalization - essential for Indic scripts
    ])
    
    # CRITICAL: Use Whitespace pre-tokenizer for Indic scripts
    # ByteLevel breaks characters into UTF-8 bytes which is wrong for Kannada
    # Whitespace preserves Kannada characters and allows BPE to learn subwords
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Set up the trainer with appropriate parameters
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,  # Lower this if vocab is too small
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    
    # Train the tokenizer
    print("\nTraining tokenizer...")
    print("This may take a few minutes...")
    tokenizer.train([corpus_file], trainer)
    
    # Add post-processor for special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"\n✓ Tokenizer saved to {tokenizer_path}")
    
    # Get actual vocabulary size
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"✓ Vocabulary size: {actual_vocab_size}")
    
    # Check if we met the requirement
    if actual_vocab_size > 5000:
        print(f"✅ Vocabulary size requirement MET: {actual_vocab_size} > 5000")
    else:
        print(f"⚠️  Vocabulary size requirement NOT MET: {actual_vocab_size} <= 5000")
        print(f"💡 Try: python train_bpe.py --vocab-size {vocab_size + 5000} --min-frequency 1")
    
    # Save metadata
    metadata = {
        "vocab_size": actual_vocab_size,
        "corpus_file": corpus_file,
        "min_frequency": min_frequency,
        "language": "Kannada (kn)",
        "pre_tokenizer": "Whitespace",
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metadata saved to {metadata_path}")
    
    return tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for Kannada")
    parser.add_argument(
        "--corpus",
        type=str,
        default="kannada_corpus.txt",
        help="Path to Kannada corpus file"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Target vocabulary size (must be >5000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="kannada_tokenizer",
        help="Output directory for the trained tokenizer"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,  # Changed default to 1 for better vocab coverage
        help="Minimum frequency for tokens"
    )
    
    args = parser.parse_args()
    
    tokenizer = train_kannada_bpe(
        corpus_file=args.corpus,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        min_frequency=args.min_frequency,
    )
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
    print("\nNext steps:")
    print("  python validate_tokenizer.py")
