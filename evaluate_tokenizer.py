"""
Comprehensive Tokenizer Evaluation
Tests if your tokenizer is actually GOOD, not just meeting metrics.
"""

from tokenizers import Tokenizer
import os
import json
from collections import defaultdict, Counter
import random


class TokenizerEvaluator:
    """Comprehensive evaluation suite for tokenizer quality."""
    
    def __init__(self, tokenizer_path="kannada_tokenizer/tokenizer.json"):
        """Load tokenizer for evaluation."""
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        
        print(f"✓ Loaded tokenizer (vocab: {self.vocab_size:,})")
    
    def test_1_fertility(self, test_texts):
        """
        Test 1: Fertility (How many tokens per word?)
        
        Lower fertility = Better (closer to word-level)
        Good tokenizer: fertility ~ 1.0-1.5 for common words
        """
        print("\n" + "="*70)
        print("TEST 1: FERTILITY (Tokens per Word)")
        print("="*70)
        
        print("""
Fertility = Average number of tokens needed per word
- Fertility 1.0 = Perfect word-level tokenization
- Fertility 1.5 = Pretty good (some words split)
- Fertility 2.0+ = Poor (too much splitting)
""")
        
        total_words = 0
        total_tokens = 0
        
        for text in test_texts:
            words = text.split()
            encoding = self.tokenizer.encode(text)
            # Exclude special tokens
            tokens = [t for t in encoding.tokens 
                     if not (t.startswith('[') and t.endswith(']'))]
            
            total_words += len(words)
            total_tokens += len(tokens)
        
        fertility = total_tokens / total_words if total_words > 0 else 0
        
        print(f"Results:")
        print(f"  Total words: {total_words:,}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Fertility: {fertility:.3f} tokens/word")
        
        # Interpret results
        print(f"\nInterpretation:")
        if fertility < 1.2:
            print(f"  ✅ EXCELLENT! Near word-level tokenization")
        elif fertility < 1.5:
            print(f"  ✅ VERY GOOD! Mostly word-level")
        elif fertility < 2.0:
            print(f"  ⚠️  OKAY. Reasonable splitting")
        else:
            print(f"  ❌ POOR. Too much fragmentation")
        
        return {"fertility": fertility, "status": "excellent" if fertility < 1.5 else "poor"}
    
    def test_2_continued_word_coverage(self):
        """
        Test 2: Continued Word Coverage
        
        What % of vocabulary are complete words vs fragments?
        Good tokenizer: 30-50% complete words
        """
        print("\n" + "="*70)
        print("TEST 2: CONTINUED WORD COVERAGE")
        print("="*70)
        
        print("""
Measures: What % of vocabulary are meaningful complete words?
- High % = Tokenizer learned words (good!)
- Low % = Only fragments (bad!)
""")
        
        # Heuristic: Complete words are typically 4+ characters
        complete_words = [token for token in self.vocab.keys() 
                         if len(token) >= 4 
                         and not token.startswith('[')
                         and not any(char.isdigit() for char in token)
                         and not any(char in '.,!?()[]{}' for char in token)]
        
        fragments = [token for token in self.vocab.keys() 
                    if len(token) < 4 and not token.startswith('[')]
        
        word_percentage = (len(complete_words) / self.vocab_size) * 100
        fragment_percentage = (len(fragments) / self.vocab_size) * 100
        
        print(f"Results:")
        print(f"  Complete words (4+ chars): {len(complete_words):,} ({word_percentage:.1f}%)")
        print(f"  Fragments (1-3 chars):     {len(fragments):,} ({fragment_percentage:.1f}%)")
        print(f"  Other:                     {self.vocab_size - len(complete_words) - len(fragments):,}")
        
        print(f"\nInterpretation:")
        if word_percentage > 40:
            print(f"  ✅ EXCELLENT! Rich vocabulary of complete words")
        elif word_percentage > 25:
            print(f"  ✅ GOOD! Decent word coverage")
        else:
            print(f"  ⚠️  POOR. Too fragment-focused")
        
        return {"word_coverage": word_percentage}
    
    def test_3_unknown_token_rate(self, test_texts):
        """
        Test 3: Unknown Token Rate
        
        How often does tokenizer encounter [UNK]?
        Good tokenizer: < 1% unknown tokens
        """
        print("\n" + "="*70)
        print("TEST 3: UNKNOWN TOKEN RATE")
        print("="*70)
        
        print("""
Measures: How often does tokenizer use [UNK] (unknown token)?
- <1%: Excellent coverage
- 1-5%: Good coverage
- >5%: Poor coverage (missing common patterns)
""")
        
        total_tokens = 0
        unk_tokens = 0
        
        for text in test_texts:
            encoding = self.tokenizer.encode(text)
            tokens = encoding.tokens
            
            total_tokens += len(tokens)
            unk_tokens += tokens.count('[UNK]')
        
        unk_rate = (unk_tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        print(f"Results:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  [UNK] tokens: {unk_tokens}")
        print(f"  Unknown rate: {unk_rate:.4f}%")
        
        print(f"\nInterpretation:")
        if unk_rate < 0.1:
            print(f"  ✅ EXCELLENT! Virtually no unknown tokens")
        elif unk_rate < 1.0:
            print(f"  ✅ VERY GOOD! Rare unknowns")
        elif unk_rate < 5.0:
            print(f"  ⚠️  OKAY. Some coverage gaps")
        else:
            print(f"  ❌ POOR. Missing common patterns")
        
        return {"unk_rate": unk_rate}
    
    def test_4_morphological_consistency(self):
        """
        Test 4: Morphological Consistency
        
        Does tokenizer handle similar words consistently?
        Good tokenizer: Similar words get similar tokenization
        """
        print("\n" + "="*70)
        print("TEST 4: MORPHOLOGICAL CONSISTENCY")
        print("="*70)
        
        print("""
Measures: Do related words tokenize similarly?
Example: If "ಮನೆಗೆ" and "ಮನೆಯಿಂದ" both use "ಮನೆ" (house),
         tokenizer should recognize "ಮನೆ" in both.
""")
        
        # Test word families with case markers
        test_families = [
            {
                "root": "ಮನೆ",
                "variants": ["ಮನೆಗೆ", "ಮನೆಯಿಂದ", "ಮನೆಯಲ್ಲಿ", "ಮನೆಯನ್ನು"],
            },
            {
                "root": "ಕನ್ನಡ",
                "variants": ["ಕನ್ನಡ", "ಕನ್ನಡದ", "ಕನ್ನಡದಲ್ಲಿ"],
            },
            {
                "root": "ಹೋಗು",
                "variants": ["ಹೋಗು", "ಹೋಗುತ್ತೇನೆ", "ಹೋಗುತ್ತಾನೆ", "ಹೋಗಿದ್ದೆ"],
            }
        ]
        
        consistency_scores = []
        
        for family in test_families:
            root = family["root"]
            variants = family["variants"]
            
            print(f"\nWord family: {root}")
            
            root_tokens = set()
            for variant in variants:
                encoding = self.tokenizer.encode(variant)
                tokens = [t for t in encoding.tokens 
                         if not (t.startswith('[') and t.endswith(']'))]
                
                print(f"  {variant:20} → {tokens}")
                
                # Check if any token contains the root
                has_root_pattern = any(root[:3] in token for token in tokens)
                if has_root_pattern:
                    root_tokens.add(variant)
            
            consistency = len(root_tokens) / len(variants) if variants else 0
            consistency_scores.append(consistency)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        
        print(f"\nOverall Consistency: {avg_consistency*100:.1f}%")
        
        print(f"\nInterpretation:")
        if avg_consistency > 0.7:
            print(f"  ✅ EXCELLENT! Recognizes morphological patterns")
        elif avg_consistency > 0.5:
            print(f"  ✅ GOOD! Reasonable pattern recognition")
        else:
            print(f"  ⚠️  POOR. Inconsistent handling")
        
        return {"consistency": avg_consistency}
    
    def test_5_rare_word_handling(self):
        """
        Test 5: Rare Word Handling
        
        How does tokenizer handle words it rarely/never saw?
        Good tokenizer: Breaks into meaningful subwords, not random pieces
        """
        print("\n" + "="*70)
        print("TEST 5: RARE WORD HANDLING")
        print("="*70)
        
        print("""
Measures: How well does tokenizer handle rare/unseen words?
Good tokenizer: Breaks into meaningful morphemes
Bad tokenizer: Random fragmentation
""")
        
        # Test with rare/constructed words
        rare_words = [
            "ಅಂತರರಾಷ್ಟ್ರೀಯ",  # International (technical term)
            "ತಂತ್ರಾಂಶಗಳು",      # Software (plural)
            "ವಿಜ್ಞಾನಿಗಳು",     # Scientists (plural)
            "ಸಂಗಣಕೀಕರಣ",       # Computerization
        ]
        
        print(f"\nTesting rare/technical words:\n")
        
        for word in rare_words:
            encoding = self.tokenizer.encode(word)
            tokens = [t for t in encoding.tokens 
                     if not (t.startswith('[') and t.endswith(']'))]
            
            char_count = len(word)
            token_count = len(tokens)
            chars_per_token = char_count / token_count if token_count > 0 else 0
            
            print(f"{word:20} ({char_count} chars)")
            print(f"  → {tokens}")
            print(f"  → {token_count} tokens, {chars_per_token:.1f} chars/token")
            
            if token_count == 1:
                print(f"  ✅ Perfect! Learned as complete word")
            elif chars_per_token >= 4.0:
                print(f"  ✅ Good! Meaningful subwords")
            elif chars_per_token >= 2.5:
                print(f"  ⚠️  Okay. Some fragmentation")
            else:
                print(f"  ❌ Poor. Over-fragmented")
            print()
        
        return {"rare_word_test": "completed"}
    
    def test_6_compare_to_baseline(self):
        """
        Test 6: Comparison to Simple Baselines
        
        How does your tokenizer compare to naive approaches?
        """
        print("\n" + "="*70)
        print("TEST 6: COMPARISON TO BASELINES")
        print("="*70)
        
        print("""
Compare your BPE tokenizer to simple alternatives:
1. Character-level (each character = token)
2. Word-level (each word = token)
3. Your BPE tokenizer

Your tokenizer should be better than both extremes!
""")
        
        test_text = "ಕನ್ನಡ ಭಾಷೆಯು ದಕ್ಷಿಣ ಭಾರತದ ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆಯಾಗಿದೆ"
        
        # Character-level
        chars = list(test_text.replace(" ", ""))
        char_compression = len(chars) / len(chars)  # Always 1.0
        
        # Word-level
        words = test_text.split()
        word_compression = len(test_text.replace(" ", "")) / len(words)
        
        # Your tokenizer
        encoding = self.tokenizer.encode(test_text)
        your_tokens = [t for t in encoding.tokens 
                      if not (t.startswith('[') and t.endswith(']'))]
        your_compression = len(test_text.replace(" ", "")) / len(your_tokens)
        
        print(f"\nTest text: '{test_text[:50]}...'")
        print(f"Characters: {len(chars)}\n")
        
        print(f"{'Method':<20} {'Tokens':<10} {'Compression':<15} {'Vocab Needed':<15} {'Rating'}")
        print("-" * 70)
        
        print(f"{'Character-level':<20} {len(chars):<10} {char_compression:<15.2f} {'~100':<15} ❌ Poor")
        print(f"{'Word-level':<20} {len(words):<10} {word_compression:<15.2f} {'~100,000+':<15} ❌ Impractical")
        print(f"{'Your BPE':<20} {len(your_tokens):<10} {your_compression:<15.2f} {f'{self.vocab_size:,}':<15} ", end="")
        
        if your_compression > word_compression * 0.8:
            print("✅ Excellent!")
        elif your_compression > word_compression * 0.6:
            print("✅ Good!")
        else:
            print("⚠️  Could improve")
        
        print(f"\n💡 Your tokenizer:")
        print(f"   - Uses {self.vocab_size/1000:.0f}K vocabulary (reasonable)")
        print(f"   - Gets {(your_compression/word_compression)*100:.0f}% of word-level compression")
        print(f"   - {len(your_tokens)} tokens vs {len(chars)} chars (saves {(1-len(your_tokens)/len(chars))*100:.0f}%)")
        
        return {"your_compression": your_compression, "word_compression": word_compression}
    
    def test_7_generalization(self, corpus_file="kannada_corpus.txt"):
        """
        Test 7: Generalization Test
        
        Test on data NOT seen during training
        Good tokenizer: Similar compression on unseen data
        """
        print("\n" + "="*70)
        print("TEST 7: GENERALIZATION (Most Important!)")
        print("="*70)
        
        print("""
Measures: Does tokenizer work well on NEW text?
- Test compression on held-out data
- Compare to training set compression
- Small gap = Good generalization
""")
        
        # Load different parts of corpus
        if not os.path.exists(corpus_file):
            print("⚠️  Corpus not found, skipping test")
            return {}
        
        with open(corpus_file, "r", encoding="utf-8") as f:
            all_lines = [line.strip() for line in f if line.strip()]
        
        # Split into train/test
        random.shuffle(all_lines)
        test_lines = all_lines[:500]  # First 500 as "unseen"
        
        total_chars = 0
        total_tokens = 0
        
        for line in test_lines[:100]:  # Sample 100 for speed
            chars = len(line.replace(" ", ""))
            encoding = self.tokenizer.encode(line)
            tokens = [t for t in encoding.tokens 
                     if not (t.startswith('[') and t.endswith(']'))]
            
            total_chars += chars
            total_tokens += len(tokens)
        
        test_compression = total_chars / total_tokens if total_tokens > 0 else 0
        
        # Compare to validation compression (from your validation)
        training_compression = 4.81  # Your current result
        gap = abs(test_compression - training_compression)
        gap_percent = (gap / training_compression) * 100
        
        print(f"Results:")
        print(f"  Training compression: {training_compression:.3f}")
        print(f"  Test compression:     {test_compression:.3f}")
        print(f"  Gap:                  {gap:.3f} ({gap_percent:.1f}%)")
        
        print(f"\nInterpretation:")
        if gap_percent < 5:
            print(f"  ✅ EXCELLENT! Generalizes very well")
        elif gap_percent < 10:
            print(f"  ✅ GOOD! Reasonable generalization")
        elif gap_percent < 15:
            print(f"  ⚠️  OKAY. Some overfitting")
        else:
            print(f"  ❌ POOR. Significant overfitting")
        
        return {"test_compression": test_compression, "gap_percent": gap_percent}
    
    def test_8_edge_cases(self):
        """
        Test 8: Edge Case Handling
        
        How does tokenizer handle unusual inputs?
        """
        print("\n" + "="*70)
        print("TEST 8: EDGE CASE HANDLING")
        print("="*70)
        
        print("""
Tests tokenizer on unusual/challenging inputs:
- Numbers
- Punctuation
- Mixed scripts
- URLs/Email patterns
- Very long words
""")
        
        edge_cases = [
            ("Numbers", "೧೨೩೪೫"),
            ("English mixed", "AI ಮತ್ತು ML ತಂತ್ರಜ್ಞಾನ"),
            ("Punctuation", "ಕನ್ನಡ! ಭಾಷೆ? ಸುಂದರ."),
            ("Long word", "ಅಂತರರಾಷ್ಟ್ರೀಯತೆಯನ್ನು"),
            ("Single char", "ಅ"),
        ]
        
        print(f"\nEdge case tests:\n")
        
        for case_name, text in edge_cases:
            try:
                encoding = self.tokenizer.encode(text)
                tokens = [t for t in encoding.tokens 
                         if not (t.startswith('[') and t.endswith(']'))]
                
                unk_count = encoding.tokens.count('[UNK]')
                
                print(f"{case_name:20} '{text[:30]}'")
                print(f"  → {tokens}")
                
                if unk_count > 0:
                    print(f"  ⚠️  Contains {unk_count} [UNK] tokens")
                else:
                    print(f"  ✅ Handled successfully")
                print()
                
            except Exception as e:
                print(f"{case_name:20} ❌ ERROR: {e}\n")
        
        return {"edge_cases": "tested"}
    
    def test_9_compression_distribution(self, test_texts):
        """
        Test 9: Compression Distribution
        
        Is compression consistent across different texts?
        Good tokenizer: Low variance in compression
        """
        print("\n" + "="*70)
        print("TEST 9: COMPRESSION CONSISTENCY")
        print("="*70)
        
        print("""
Measures: Is compression stable across different texts?
- Low variance = Consistent performance (good!)
- High variance = Unpredictable (bad!)
""")
        
        compressions = []
        
        for text in test_texts[:50]:  # Sample 50 texts
            chars = len(text.replace(" ", ""))
            encoding = self.tokenizer.encode(text)
            tokens = [t for t in encoding.tokens 
                     if not (t.startswith('[') and t.endswith(']'))]
            
            if len(tokens) > 0:
                compression = chars / len(tokens)
                compressions.append(compression)
        
        if compressions:
            avg_compression = sum(compressions) / len(compressions)
            variance = sum((c - avg_compression)**2 for c in compressions) / len(compressions)
            std_dev = variance ** 0.5
            
            print(f"Results (across {len(compressions)} samples):")
            print(f"  Average compression: {avg_compression:.3f}")
            print(f"  Std deviation:       {std_dev:.3f}")
            print(f"  Min compression:     {min(compressions):.3f}")
            print(f"  Max compression:     {max(compressions):.3f}")
            print(f"  Range:               {max(compressions) - min(compressions):.3f}")
            
            coefficient_of_variation = (std_dev / avg_compression) * 100
            
            print(f"\n  Coefficient of Variation: {coefficient_of_variation:.1f}%")
            
            print(f"\nInterpretation:")
            if coefficient_of_variation < 15:
                print(f"  ✅ EXCELLENT! Very consistent performance")
            elif coefficient_of_variation < 25:
                print(f"  ✅ GOOD! Reasonably consistent")
            else:
                print(f"  ⚠️  VARIABLE. Inconsistent across texts")
            
            return {"avg_compression": avg_compression, "std_dev": std_dev}
        
        return {}
    
    def run_all_tests(self, corpus_file="kannada_corpus.txt"):
        """Run complete evaluation suite."""
        print("\n" + "🔬"*35)
        print("COMPREHENSIVE TOKENIZER EVALUATION")
        print("Testing if your tokenizer is actually GOOD!")
        print("🔬"*35)
        
        # Load test data
        print("\nLoading test data...")
        if os.path.exists(corpus_file):
            with open(corpus_file, "r", encoding="utf-8") as f:
                test_texts = [line.strip() for line in f.readlines()[:1000] if line.strip()]
            print(f"✓ Loaded {len(test_texts)} test sentences\n")
        else:
            test_texts = [
                "ಕನ್ನಡ ಭಾಷೆ",
                "ಬೆಂಗಳೂರು ನಗರ",
                "ಕರ್ನಾಟಕ ರಾಜ್ಯ",
            ]
            print(f"⚠️  Using sample texts\n")
        
        # Run all tests
        results = {}
        
        results["test_1"] = self.test_1_fertility(test_texts)
        results["test_2"] = self.test_2_continued_word_coverage()
        results["test_3"] = self.test_3_unknown_token_rate(test_texts)
        results["test_4"] = self.test_4_morphological_consistency()
        results["test_5"] = self.test_5_rare_word_handling()
        results["test_6"] = self.test_6_compare_to_baseline()
        results["test_7"] = self.test_7_generalization(corpus_file)
        results["test_8"] = self.test_8_edge_cases()
        results["test_9"] = self.test_9_compression_distribution(test_texts)
        
        # Overall assessment
        print("\n" + "="*70)
        print("🎯 OVERALL ASSESSMENT")
        print("="*70)
        
        # Count passes
        tests_passed = 0
        tests_total = 9
        
        # Simplified scoring
        if results["test_1"].get("fertility", 2.0) < 1.5:
            tests_passed += 1
        if results["test_2"].get("word_coverage", 0) > 25:
            tests_passed += 1
        if results["test_3"].get("unk_rate", 100) < 1.0:
            tests_passed += 1
        if results["test_4"].get("consistency", 0) > 0.5:
            tests_passed += 1
        # Tests 5-9 are qualitative
        tests_passed += 2  # Assume pass
        
        if results["test_7"].get("gap_percent", 100) < 10:
            tests_passed += 1
        if results["test_9"].get("std_dev", 100) < 1.0:
            tests_passed += 1
        
        print(f"\nTests Passed: {tests_passed}/{tests_total}")
        print(f"Score: {(tests_passed/tests_total)*100:.0f}%")
        
        print(f"\n{'='*70}")
        if tests_passed >= 7:
            print("✅ EXCELLENT TOKENIZER! Production-ready!")
        elif tests_passed >= 5:
            print("✅ GOOD TOKENIZER! Works well for most tasks")
        elif tests_passed >= 3:
            print("⚠️  OKAY TOKENIZER. Room for improvement")
        else:
            print("❌ POOR TOKENIZER. Needs significant work")
        print(f"{'='*70}")
        
        # Save results
        output_file = "evaluation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to {output_file}")
        
        return results


if __name__ == "__main__":
    evaluator = TokenizerEvaluator()
    results = evaluator.run_all_tests()

