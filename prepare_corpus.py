"""
Script to download and prepare Kannada text corpus for BPE training.
Downloads real Kannada text from online sources.
"""

import os
import random
from datasets import load_dataset
from tqdm import tqdm


def download_kannada_corpus(output_file="kannada_corpus.txt", num_samples=100000):
    """
    Download real Kannada text from online datasets.
    Tries multiple sources until one works.
    
    Args:
        output_file: Path to save the corpus
        num_samples: Number of samples to download
    """
    print("Attempting to download Kannada corpus from online sources...")
    print(f"Target: {num_samples} samples\n")
    
    # List of datasets to try (in order of preference)
    datasets_to_try = [
        {
            "name": "oscar-corpus/OSCAR-2301",
            "config": "kn",
            "split": "train",
            "text_field": "text",
            "description": "OSCAR - Large web-crawled multilingual corpus"
        },
        {
            "name": "mc4",
            "config": "kn",
            "split": "train",
            "text_field": "text",
            "description": "mC4 - Multilingual C4 dataset"
        },
        {
            "name": "facebook/flores",
            "config": "kan_Knda",
            "split": "dev",
            "text_field": "sentence",
            "description": "FLORES - Translation dataset"
        },
        {
            "name": "wikimedia/wikipedia",
            "config": "20231101.kn",
            "split": "train",
            "text_field": "text",
            "description": "Wikipedia - Kannada Wikipedia dump"
        },
    ]
    
    for dataset_info in datasets_to_try:
        try:
            print(f"{'='*60}")
            print(f"Trying: {dataset_info['description']}")
            print(f"Dataset: {dataset_info['name']}")
            print(f"{'='*60}\n")
            
            # Load dataset in streaming mode for large datasets
            dataset = load_dataset(
                dataset_info["name"],
                dataset_info["config"],
                split=dataset_info["split"],
                streaming=True,
                trust_remote_code=False
            )
            
            print(f"✓ Dataset loaded successfully!")
            print(f"Collecting {num_samples} samples...\n")
            
            with open(output_file, "w", encoding="utf-8") as f:
                count = 0
                text_field = dataset_info["text_field"]
                
                for example in tqdm(dataset, desc="Downloading", total=num_samples):
                    text = example.get(text_field, "")
                    
                    if isinstance(text, str):
                        text = text.strip()
                    else:
                        continue
                    
                    # Only keep substantial text
                    if text and len(text) > 20:
                        # Split long texts into sentences for better training
                        if len(text) > 200:
                            # Try to split by common sentence delimiters
                            for delimiter in ["।", ".", "!", "?"]:
                                if delimiter in text:
                                    sentences = text.split(delimiter)
                                    for sent in sentences:
                                        sent = sent.strip()
                                        if len(sent) > 20:
                                            f.write(sent + "\n")
                                            count += 1
                                            if count >= num_samples:
                                                break
                                    break
                            else:
                                f.write(text + "\n")
                                count += 1
                        else:
                            f.write(text + "\n")
                            count += 1
                        
                        if count >= num_samples:
                            break
            
            if count > 0:
                print(f"\n{'='*60}")
                print(f"✅ SUCCESS!")
                print(f"{'='*60}")
                print(f"✓ Downloaded {count:,} text samples")
                file_size = os.path.getsize(output_file) / (1024 * 1024)
                print(f"✓ Corpus size: {file_size:.2f} MB")
                print(f"✓ Source: {dataset_info['description']}")
                print(f"✓ Saved to: {output_file}")
                return output_file
            
        except Exception as e:
            print(f"❌ Failed: {e}\n")
            continue
    
    # If all downloads fail, create a fallback corpus
    print("\n" + "="*60)
    print("⚠️  All online sources failed")
    print("="*60)
    print("Creating fallback corpus with diverse Kannada vocabulary...\n")
    create_fallback_corpus(output_file)
    return output_file


def create_fallback_corpus(output_file="kannada_corpus.txt"):
    """
    Create a fallback corpus with diverse Kannada vocabulary.
    Uses comprehensive word lists covering many topics.
    """
    print(f"Creating fallback Kannada corpus...")
    
    # Comprehensive Kannada vocabulary
    vocabulary = {
        "common": [
            "ಅವನು", "ಅವಳು", "ಅವರು", "ನಾನು", "ನಾವು", "ನೀನು", "ನೀವು",
            "ಇವನು", "ಇವಳು", "ಇವರು", "ಯಾರು", "ಏನು", "ಎಲ್ಲಿ", "ಎಷ್ಟು",
            "ಹೇಗೆ", "ಯಾವಾಗ", "ಯಾಕೆ", "ಹೌದು", "ಇಲ್ಲ", "ಹೋಗು", "ಬರು",
            "ಮಾಡು", "ತಿನ್ನು", "ಕುಡಿ", "ನೋಡು", "ಕೇಳು", "ಹೇಳು", "ಓದು",
            "ಬರೆ", "ಕೊಡು", "ತೆಗೆ", "ಹಾಕು", "ಇಡು", "ನಿಲ್ಲು", "ಕುಳಿತುಕೊಳ್ಳು",
        ],
        "body": [
            "ತಲೆ", "ಕೂದಲು", "ಮುಖ", "ಕಣ್ಣು", "ಕಿವಿ", "ಮೂಗು", "ಬಾಯಿ",
            "ಹಲ್ಲು", "ನಾಲಗೆ", "ಕೆನ್ನೆ", "ಕುತ್ತಿಗೆ", "ಭುಜ", "ಎದೆ",
            "ಹೊಟ್ಟೆ", "ಬೆನ್ನು", "ಕೈ", "ಬೆರಳು", "ಕಾಲು", "ಮೊಣಕಾಲು", "ಪಾದ"
        ],
        "animals": [
            "ನಾಯಿ", "ಬೆಕ್ಕು", "ಹಸು", "ಎತ್ತು", "ಎಮ್ಮೆ", "ಆಡು", "ಕುರಿ",
            "ಕುದುರೆ", "ಕತ್ತೆ", "ಹಂದಿ", "ಕೋಳಿ", "ಬಾತುಕೋಳಿ", "ಹಕ್ಕಿ",
            "ಗಿಳಿ", "ನವಿಲು", "ಕಾಗೆ", "ಗೂಬೆ", "ಹದ್ದು", "ಕೊಕ್ಕರೆ",
            "ಆನೆ", "ಸಿಂಹ", "ಹುಲಿ", "ಚಿರತೆ", "ಕರಡಿ", "ತೋಳ", "ನರಿ",
            "ಮೊಲ", "ಜಿಂಕೆ", "ಕಾಡುಹಂದಿ", "ಕೋತಿ", "ಅಳಿಲು",
            "ಹಾವು", "ಚೇಳು", "ಜಿಗಣೆ", "ಜೇನುಹುಳು", "ಚಿಟ್ಟೆ", "ಇರುವೆ",
            "ಮೀನು", "ಏಡಿ", "ತಿಮಿಂಗಿಲ", "ಡಾಲ್ಫಿನ್", "ಶಾರ್ಕ್"
        ],
        "food": [
            "ಅನ್ನ", "ರೊಟ್ಟಿ", "ದೋಸೆ", "ಇಡ್ಲಿ", "ವಡೆ", "ಉಪ್ಪಿಟ್ಟು",
            "ಸಾರು", "ಸಂಬಾರ", "ರಸಂ", "ಕೂಟು", "ಪಲ್ಯ", "ಕಾಳು",
            "ಹಾಲು", "ಮೊಸರು", "ಬೆಣ್ಣೆ", "ತುಪ್ಪ", "ಎಣ್ಣೆ", "ಉಪ್ಪು",
            "ಸಕ್ಕರೆ", "ಬೆಲ್ಲ", "ಮಿರ್ಚಿ", "ಅರಿಶಿನ", "ಜೀರಿಗೆ",
            "ಬಾಳೆಹಣ್ಣು", "ಆಮ್ಬೆ", "ಸೀಂಬೆ", "ಹಲಸು", "ಹಪ್ಪಳ",
            "ಕಬ್ಬು", "ಎಲುಮಿಚ್ಚೆ", "ಬೇವು", "ಬೆಳ್ಳುಳ್ಳಿ", "ಈರುಳ್ಳಿ",
            "ಬದನೆಕಾಯಿ", "ಟೊಮೇಟೊ", "ಆಲೂಗೆಡ್ಡೆ", "ಕಾಳುಗೆಡ್ಡೆ"
        ],
        "places": [
            "ಬೆಂಗಳೂರು", "ಮೈಸೂರು", "ಮಂಗಳೂರು", "ಹುಬ್ಬಳ್ಳಿ", "ಬೆಳಗಾವಿ",
            "ಕಲಬುರಗಿ", "ಧಾರವಾಡ", "ಶಿವಮೊಗ್ಗ", "ತುಮಕೂರು", "ಚಿತ್ರದುರ್ಗ",
            "ಹಂಪಿ", "ಕೂರ್ಗ್", "ಉಡುಪಿ", "ಹಾಸನ", "ಬೀದರ್",
            "ಕರ್ನಾಟಕ", "ಭಾರತ", "ದೇಶ", "ರಾಜ್ಯ", "ನಗರ", "ಗ್ರಾಮ"
        ],
        "numbers": [
            "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು",
            "ಎಂಟು", "ಒಂಬತ್ತು", "ಹತ್ತು", "ನೂರು", "ಸಾವಿರ", "ಲಕ್ಷ"
        ]
    }
    
    all_words = []
    for category, words in vocabulary.items():
        all_words.extend(words)
    
    print(f"✓ Loaded {len(all_words)} unique words")
    
    # Generate diverse sentences
    sentences = []
    
    # Create random combinations
    for _ in range(20000):
        sent_len = random.randint(3, 10)
        words = random.sample(all_words, min(sent_len, len(all_words)))
        sentences.append(" ".join(words) + ".")
    
    print(f"✓ Generated {len(sentences)} sentences")
    
    # Write to file with repetition
    with open(output_file, "w", encoding="utf-8") as f:
        for _ in range(15):  # Repeat for more training data
            random.shuffle(sentences)
            for sent in sentences:
                f.write(sent + "\n")
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Created corpus: {file_size:.2f} MB")
    print(f"✓ Unique vocabulary: {len(all_words)} words")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Kannada corpus for BPE training")
    parser.add_argument(
        "--output",
        type=str,
        default="kannada_corpus.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="Number of samples to download (more = better tokenizer)"
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip download and create fallback corpus"
    )
    
    args = parser.parse_args()
    
    if args.fallback_only:
        create_fallback_corpus(args.output)
    else:
        download_kannada_corpus(args.output, args.samples)
