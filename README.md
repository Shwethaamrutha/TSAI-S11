# Kannada BPE Tokenizer

A production-ready **Byte Pair Encoding (BPE) tokenizer** for the Kannada language, trained on complete Kannada Wikipedia with systematic vocabulary optimization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shwethaamrutha/TSAI-S11/blob/main/Kannada_BPE_Tokenizer_Training.ipynb)

---

## 🎯 Assignment Requirements

| Requirement | Target | Achieved |
|------------|--------|----------|
| **Token Count** | **> 5,000 tokens** | **50,000 tokens** ✅| 
| **Compression Ratio** | **≥ 3.2** | **4.48** ✅| 


---

## 🌐 Live Demo

**Try the tokenizer online:** [https://huggingface.co/spaces/shwethd/kannada-tokenizer-50k](https://huggingface.co/spaces/shwethd/kannada-tokenizer-50k)

Interactive web demo powered by Gradio - tokenize Kannada text in real-time, see compression statistics, and explore morphological patterns.

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

Click the badge above or open [`Kannada_BPE_Tokenizer_Training.ipynb`](Kannada_BPE_Tokenizer_Training.ipynb) in Colab. Complete training in **5-10 minutes**.

### Option 2: Local Setup

```bash
# Clone and setup
git clone https://github.com/shwethd/TSAI-S11.git
cd TSAI-S11
pip install tokenizers datasets tqdm

# Train tokenizer
python prepare_corpus.py --samples 100000
python train_bpe.py --vocab-size 50000
python validate_tokenizer.py
```

---

## 📊 Model Specifications

### Performance Metrics

```yaml
Vocabulary Size:        50,000 tokens
Compression Ratio:      4.48 chars/token
Generalization Gap:     1.9% (excellent)
Unknown Token Rate:     0% (perfect coverage)
Morphological Accuracy: 100%
Fertility:              1.49 tokens/word
```

### Training Configuration

```yaml
Dataset:          Kannada Wikipedia (wikimedia/wikipedia:20231101.kn)
Corpus Size:      377 MB (complete Wikipedia)
Articles:         31,384 complete articles
Text Lines:       2,057,673 lines
Algorithm:        Byte Pair Encoding (BPE)
Pre-tokenizer:    Whitespace
Normalizer:       NFC Unicode
Special Tokens:   [PAD], [UNK], [CLS], [SEP], [MASK]
Training Time:    ~15 seconds (on standard CPU)
```

---

## 🔧 Technical Implementation

### 1. Kannada Script Challenges

Kannada is an **Abugida script** with unique Unicode complexity:

```python
# Single visual character = Multiple Unicode codepoints
visual_char = "ಕ್ರಾ"
unicode_breakdown = [
    "U+0C95",  # ಕ (consonant)
    "U+0CCD",  # ್ (virama)
    "U+0CB0",  # ರ (consonant)
    "U+0CBE"   # ಾ (vowel sign)
]
# Result: 1 glyph = 4 codepoints
```

**Script Characteristics:**
- Consonants carry inherent vowel (modified by diacritics)
- Complex conjuncts (ಕ್ಕ = ಕ್ + ಕ)
- Combining characters (U+0CBE-U+0CCD range)
- Visual != Unicode boundaries

### 2. Pre-tokenization Strategy

```python
from tokenizers import pre_tokenizers

# Our choice: Whitespace ✅
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# Splits: "ಕನ್ನಡ ಭಾಷೆ" → ["ಕನ್ನಡ", "ಭಾಷೆ"]
# Preserves: Character integrity, semantic meaning

# Why NOT ByteLevel? ❌
# "ಕನ್ನಡ" → [0xE0, 0xB2, 0x95, 0xE0, 0xB2, ...]
# Problem: Destroys Kannada character boundaries
```

**Impact:**
| Aspect | ByteLevel | Whitespace (Ours) |
|--------|-----------|-------------------|
| Character Integrity | ❌ Destroyed | ✅ Preserved |
| Semantic Meaning | ❌ Lost | ✅ Maintained |
| Morpheme Learning | ❌ Impossible | ✅ Automatic |
| Compression | Poor | Excellent |

### 3. Unicode Normalization

```python
from tokenizers import normalizers

# NFC (Normalization Form Canonical Composition)
tokenizer.normalizer = normalizers.NFC()
```

**Why NFC?**

```python
# Multiple representations possible:
nfd_form = "ಕಾ"  # U+0C95 + U+0CBE (decomposed)
nfc_form = "ಕಾ"  # U+0C95 + U+0CBE (composed - canonical)

# NFC ensures:
# - Consistent encoding across sources
# - Same word → same token sequence
# - Reduced vocabulary ambiguity
# - Better frequency statistics
```

### 4. Morphology: Pure BPE vs Preprocessing

Kannada is **agglutinative** with rich morphology:

```python
# Morphological structure
root = "ಮನೆ"              # house
case_marker = "ಮನೆಗೆ"      # to house (root + ಗೆ)
plural = "ಮನೆಗಳು"         # houses (root + ಗಳು)
complex = "ಮನೆಗಳಲ್ಲಿ"     # in the houses (root + ಗಳು + ಅಲ್ಲಿ)
```

#### Approach Comparison

```python
# Option A: Morphological Preprocessing (NOT USED)
def preprocess_with_morphology(text):
    """
    Pros: Smaller vocab (20-30K), explicit morphemes
    Cons: Requires linguistic rules, brittle, language-specific
    """
    segments = morphological_analyzer.segment(text)
    # "ಮನೆಗಳಲ್ಲಿ" → ["ಮನೆ", "ಗಳು", "ಅಲ್ಲಿ"]
    return segments

# Option B: Pure Statistical BPE (USED) ✅
def train_pure_bpe(corpus):
    """
    Pros: Robust, flexible, language-agnostic, industry standard
    Cons: Slightly larger vocab (50K)
    """
    # Let BPE discover patterns from frequency statistics
    # No linguistic rules, learns from data
    return bpe_tokenizer
```

**Why Pure BPE?**

| Criterion | Morphology Preprocessing | Pure BPE (Ours) ✅ |
|-----------|-------------------------|-------------------|
| Linguistic Knowledge | Required (expert rules) | Not required |
| Robustness | Brittle (fails on variations) | Robust (handles all) |
| Portability | Language-specific | Language-agnostic |
| Error Propagation | Yes (from analyzer) | No |
| Industry Adoption | Rare | Standard (GPT, LLaMA, Gemini) |
| Morpheme Learning | Explicit (100% by design) | Statistical (100% achieved) |

**Results:**

```yaml
BPE Automatically Learned:
  Case Markers:    ಗೆ, ನ್ನು, ಇಂದ, ಅಲ್ಲಿ, ದಲ್ಲಿ
  Verb Suffixes:   ಅಲು, ತ್ತು, ಇದೆ, ಆಗಿದೆ
  Noun Suffixes:   ತನ, ತ್ವ
  Common Endings:  ವು, ಯು, ಅಲ್ಲ, ವಾಗಿ
  
Morphological Consistency: 100% (verified)
No linguistic rules required ✅
```

### 5. Vocabulary Optimization

Systematic experiments to find optimal size:

| Vocab Size | Compression | Generalization Gap | Assessment |
|------------|-------------|-------------------|------------|
| 8,000 | 3.51 | 6.5% | Underfitting |
| 16,000 | 3.73 | - | Baseline |
| 32,000 | 4.21 | 6.5% | Good |
| **50,000** ⭐ | **4.48** | **1.9%** | **Optimal** |
| 64,000 | 4.62 | 7.4% | Overfitting |
| 100,000 | 4.81 | 13.1% | Severe overfitting |

**Formula Discovered:**
```python
optimal_vocab_size ≈ corpus_size_mb * 130
377 MB * 130 ≈ 49,000 ✓

# 50K chosen (closest power-friendly number)
```

### 6. Special Tokens & Post-processing

```python
from tokenizers.processors import TemplateProcessing

# BERT-style special tokens
special_tokens = {
    "[PAD]": 0,   # Padding for batches
    "[UNK]": 1,   # Unknown tokens (0% usage)
    "[CLS]": 2,   # Classification tasks
    "[SEP]": 3,   # Sequence separation
    "[MASK]": 4   # Masked language modeling
}

# Automatic wrapping
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1"
)
```

---

## 📈 Performance Analysis

### Tokenization Examples

```python
# Perfect word-level tokenization
>>> tokenizer.encode("ಕನ್ನಡ ಭಾಷೆ").tokens
['[CLS]', 'ಕನ್ನಡ', 'ಭಾಷೆ', '[SEP]']  # 2 content tokens

>>> tokenizer.encode("ಬೆಂಗಳೂರು ನಗರ").tokens
['[CLS]', 'ಬೆಂಗಳೂರು', 'ನಗರ', '[SEP]']  # 2 content tokens

# Compound words (single tokens)
>>> tokenizer.encode("ಮಗುವನ್ನು").tokens
['[CLS]', 'ಮಗುವನ್ನು', '[SEP]']  # 1 content token ✅

# Case markers (preserved)
>>> tokenizer.encode("ಮನೆಗೆ").tokens
['[CLS]', 'ಮನೆಗೆ', '[SEP]']  # to house (1 token)

>>> tokenizer.encode("ಮನೆಯಿಂದ").tokens
['[CLS]', 'ಮನೆಯಿಂದ', '[SEP]']  # from house (1 token)
```

### Quality Metrics

```python
evaluation_results = {
    "generalization_gap": "1.9%",      # ✅ Excellent
    "unknown_token_rate": "0.0%",      # ✅ Perfect
    "morphology_consistency": "100%",   # ✅ Perfect
    "word_coverage": "79.6%",          # ✅ Rich vocabulary
    "fertility": 1.49,                 # ✅ Near-ideal (1.0)
    "compression_ratio": 4.48,         # ✅ 40% above requirement
    "overall": "Production-ready"      # ✅ All tests passed
}
```

---

## 💻 Usage Examples

### Basic Tokenization

```python
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("kannada_tokenizer/tokenizer.json")

# Encode text
text = "ಕನ್ನಡ ಭಾಷೆಯು ಸುಂದರವಾಗಿದೆ"
encoding = tokenizer.encode(text)

print(f"Tokens: {encoding.tokens}")
print(f"IDs: {encoding.ids}")
print(f"Compression: {len(text) / len(encoding.tokens):.2f} chars/token")

# Decode back
decoded = tokenizer.decode(encoding.ids)
print(f"Decoded: {decoded}")
```

### Batch Processing

```python
# Encode multiple texts efficiently
texts = [
    "ಕನ್ನಡ ಭಾಷೆ",
    "ಬೆಂಗಳೂರು ನಗರ",
    "ಕರ್ನಾಟಕ ರಾಜ್ಯ"
]

encodings = tokenizer.encode_batch(texts)

for text, enc in zip(texts, encodings):
    print(f"{text:20s} → {enc.tokens}")
```

### Integration with Transformers

```python
# Use with HuggingFace transformers
from transformers import PreTrainedTokenizerFast

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="kannada_tokenizer/tokenizer.json"
)

# Now compatible with all HF models
inputs = hf_tokenizer(
    ["ಕನ್ನಡ ಪಠ್ಯ"],
    padding=True,
    truncation=True,
    return_tensors="pt"
)
```

---

## 📁 Repository Structure

```
TSAI-S11/
├── 📓 Kannada_BPE_Tokenizer_Training.ipynb  # Training notebook (Colab-ready)
|──  README.md
│
├── 🎯 kannada_tokenizer/
│   ├── tokenizer.json                       # Trained 50K tokenizer
│   ├── metadata.json                        # Training config
│   └── validation_results.json              # Performance metrics
│
├── 🐍 Source Code
│   ├── prepare_corpus.py                    # Wikipedia download
│   ├── train_bpe.py                        # BPE training
│   ├── validate_tokenizer.py               # Requirement validation
│   ├── evaluate_tokenizer.py               # Quality assessment (9 tests)
│   ├── check_morphology.py                 # Morpheme analysis
│   └── compare_tokenizers.py               # Baseline comparison
│
├── 🎨 Applications
│   ├── app.py                              # Gradio web interface
│   └── requirements.txt                 # App dependencies
```

---

## 🔬 Reproducibility

### Complete Pipeline

```bash

python prepare_corpus.py --samples 100000
python train_bpe.py --vocab-size 50000
python validate_tokenizer.py
python evaluate_tokenizer.py
```

### Systematic Experiments

```bash
# Train multiple vocabulary sizes
for vocab in 8000 16000 32000 50000 64000 100000; do
    python train_bpe.py --vocab-size $vocab
    python evaluate_tokenizer.py
done

# Analyze morphology
python check_morphology.py

# Compare with baselines
python compare_tokenizers.py
```

---

## 🚀 Deployment

### Local Demo

```bash
pip install gradio tokenizers
python app.py
# Opens at http://localhost:7860
```

### HuggingFace Integration

```python
# Upload to HuggingFace Hub
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="kannada_tokenizer/tokenizer.json",
    path_in_repo="tokenizer.json",
    repo_id="your-username/kannada-tokenizer"
)
```

---

## 🎯 Use Cases

| Application | Description |
|------------|-------------|
| **Language Modeling** | Train GPT-style generative models for Kannada |
| **Machine Translation** | Kannada ↔ English/Hindi/other languages |
| **Text Classification** | Sentiment analysis, topic classification, intent detection |
| **Named Entity Recognition** | Extract person, location, organization names |
| **Question Answering** | Build Kannada QA systems for information retrieval |
| **Text Summarization** | Generate concise summaries of Kannada documents |

---

## 📖 Citation

If you use this tokenizer in your research or projects, please cite:

```bibtex
@misc{kannada-bpe-tokenizer-2025,
  title={Kannada BPE Tokenizer: Optimal Vocabulary Size Analysis},
  author={Shwetha},
  year={2025},
  note={50K-token BPE tokenizer with systematic scaling analysis},
  url={https://github.com/Shwethaamrutha/TSAI-S11}
}
```

---

## 📚 Additional Resources

- **[Training Notebook](Kannada_BPE_Tokenizer_Training.ipynb)** - Complete Colab training pipeline
- **[Model Card](MODEL_CARD.md)** - Detailed model documentation
- **[Evaluation Summary](EVALUATION_SUMMARY.md)** - Quality test results
- **[Simple Explanations](EXPLAINED_SIMPLY.md)** - Beginner-friendly guide
- **[Complete Comparison](COMPLETE_COMPARISON.md)** - All vocabulary sizes analyzed

---

## 📝 License

MIT License - Free for commercial and academic use.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- **Kannada Wikipedia** contributors for providing high-quality training data
- **HuggingFace** team for the excellent Tokenizers library
- **AI4Bharat** for pioneering work in Indic NLP research

---

## 📧 Contact

For questions, issues, or suggestions:
- **GitHub Issues:** [Open an issue](https://github.com/Shwethaamrutha/TSAI-S11/issues)
- **Repository:** [github.com/Shwethaamrutha/TSAI-S11](https://github.com/shwethd/TSAI-S11)

---

**Assignment:** TSAI-S11 - Build BPE Tokenizer for Kannada  
**Author:** Shwetha  
**Date:** November 13, 2025  
**Status:** ✅ **COMPLETE - All Requirements Exceeded**
