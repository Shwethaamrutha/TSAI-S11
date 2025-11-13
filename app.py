"""
HuggingFace Gradio App for Kannada BPE Tokenizer
Interactive demo to showcase your tokenizer
"""

import gradio as gr
from tokenizers import Tokenizer
import os


# Load tokenizer
TOKENIZER_PATH = "kannada_tokenizer/tokenizer.json"

if os.path.exists(TOKENIZER_PATH):
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    VOCAB_SIZE = tokenizer.get_vocab_size()
else:
    raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")


def tokenize_text(text, show_details=False):
    """
    Tokenize input text and return formatted results.
    """
    if not text.strip():
        return "⚠️ Please enter some Kannada text to tokenize.", "", ""
    
    # Encode the text
    encoding = tokenizer.encode(text)
    
    # Get tokens and IDs
    all_tokens = encoding.tokens
    all_ids = encoding.ids
    
    # Separate content tokens from special tokens
    content_tokens = []
    content_ids = []
    for token, token_id in zip(all_tokens, all_ids):
        if not (token.startswith('[') and token.endswith(']')):
            content_tokens.append(token)
            content_ids.append(token_id)
    
    # Calculate statistics
    char_count = len(text.replace(" ", "").replace("\n", ""))
    token_count = len(content_tokens)
    compression = char_count / token_count if token_count > 0 else 0
    
    # Format main output (clean and professional)
    result_text = f"### 🔤 Tokenization Result\n\n"
    
    # Show tokens in a clean format
    token_display = " | ".join([f"`{t}`" for t in content_tokens])
    result_text += f"{token_display}\n\n"
    
    # Statistics in cards
    result_text += f"### 📊 Statistics\n\n"
    result_text += f"<div style='display: flex; gap: 20px; flex-wrap: wrap;'>\n"
    result_text += f"<div style='padding: 15px; background: #f0f7ff; border-radius: 8px; flex: 1; min-width: 150px;'>\n"
    result_text += f"<div style='font-size: 0.9em; color: #666;'>Characters</div>\n"
    result_text += f"<div style='font-size: 1.8em; font-weight: 600; color: #0066cc;'>{char_count}</div>\n"
    result_text += f"</div>\n"
    result_text += f"<div style='padding: 15px; background: #f0fff4; border-radius: 8px; flex: 1; min-width: 150px;'>\n"
    result_text += f"<div style='font-size: 0.9em; color: #666;'>Tokens</div>\n"
    result_text += f"<div style='font-size: 1.8em; font-weight: 600; color: #059669;'>{token_count}</div>\n"
    result_text += f"</div>\n"
    result_text += f"<div style='padding: 15px; background: #fef3f2; border-radius: 8px; flex: 1; min-width: 150px;'>\n"
    result_text += f"<div style='font-size: 0.9em; color: #666;'>Compression</div>\n"
    result_text += f"<div style='font-size: 1.8em; font-weight: 600; color: #dc2626;'>{compression:.2f}x</div>\n"
    result_text += f"</div>\n"
    result_text += f"</div>\n\n"
    
    # Details section (optional, collapsible)
    details = ""
    if show_details:
        details += f"### 🔍 Detailed View\n\n"
        details += f"**Token IDs:** `{content_ids}`\n\n"
        details += f"**Individual Tokens:**\n\n"
        for i, (token, tid) in enumerate(zip(content_tokens, content_ids), 1):
            details += f"{i}. `{token}` → ID {tid}\n"
    else:
        details = "💡 *Enable 'Show Details' to see token IDs and breakdown*"
    
    return result_text, details


def get_vocab_info():
    """Get vocabulary information."""
    vocab = tokenizer.get_vocab()
    
    info = f"### Tokenizer Information:\n\n"
    info += f"- **Vocabulary Size:** {VOCAB_SIZE:,} tokens\n"
    info += f"- **Language:** Kannada (kn)\n"
    info += f"- **Method:** Byte Pair Encoding (BPE)\n"
    info += f"- **Training Data:** Kannada Wikipedia (373 MB, 2M samples)\n"
    info += f"- **Compression Ratio:** 4.48 chars/token (average)\n"
    info += f"- **Special Tokens:** [PAD], [UNK], [CLS], [SEP], [MASK]\n\n"
    
    # Sample vocabulary
    info += f"### Sample Vocabulary (First 20 tokens):\n\n"
    for token, idx in list(vocab.items())[:20]:
        info += f"- `{token}` (ID: {idx})\n"
    
    return info


# Example texts
EXAMPLES = [
    ['ಇಲ್ಲಿ ಕೆಲವು ಸಾಮಾನ್ಯ ಕನ್ನಡ ವಾಕ್ಯಗಳಿವೆ: "ನಿಮ್ಮ ಹೆಸರೇನು?" (ನಿಮ್ಮ ಹೆಸರೇನು?), "ಏನಾಯಿತು?" (ಏನಾಯಿತು?), ಮತ್ತು "ನೀವು ಊಟ ಮಾಡಿದ್ದೀರಾ?" (ಊಟಾ ಆಯಿತ?).'],
    ["ನಾ ಚಲೋ ಅದೀನಿ, ನೀನು ಹ್ಯಾಂಗದೀರ್'ರಿ?"],
    ["ಕನ್ನಡ ಭಾಷೆ"],
    ["ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿಯಾಗಿದೆ"],
    ["ಕನ್ನಡ ಸಾಹಿತ್ಯವು ಬಹಳ ಪ್ರಾಚೀನವಾಗಿದೆ ಮತ್ತು ಸಮೃದ್ಧವಾಗಿದೆ"],
    ["ಹಂಪಿ ಕರ್ನಾಟಕದ ಒಂದು ಪ್ರಸಿದ್ಧ ಐತಿಹಾಸಿಕ ಸ್ಥಳವಾಗಿದೆ"],
    ["ಕನ್ನಡ ಭಾಷೆಯು ವಿಶ್ವದ ಪ್ರಾಚೀನ ಭಾಷೆಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ"],
]


# Create Gradio interface with professional styling
custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
        max-width: 1200px !important;
        margin: auto !important;
    }
    .markdown-text {
        font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
        font-weight: 600 !important;
    }
    .primary-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    .primary-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
    /* Card styling */
    .stat-card {
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    /* Fixed layout to prevent width changes */
    .tabs {
        width: 100% !important;
    }
    .tab-nav {
        width: 100% !important;
    }
    .tabitem {
        width: 100% !important;
        min-height: 600px !important;
    }
    /* Consistent column widths */
    .column {
        min-width: 0 !important;
    }
"""

with gr.Blocks(title="Kannada BPE Tokenizer", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(f"""
    #  Kannada BPE Tokenizer
    
    ### A production-ready tokenizer for Kannada with **{VOCAB_SIZE:,} tokens**
    
    Trained on 377 MB Kannada Wikipedia | 4.48 compression ratio | 1.9% generalization gap
    """)
    
    # Stats cards using HTML for better rendering
    gr.HTML("""
    <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); padding: 20px; border-radius: 12px; margin: 20px 0;'>
        <div style='display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;'>
            <div style='text-align: center;'>
                <div style='font-size: 2em; font-weight: 700; color: #667eea;'>50,000</div>
                <div style='font-size: 0.9em; color: #666;'>Token Vocabulary</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 2em; font-weight: 700; color: #059669;'>4.48x</div>
                <div style='font-size: 0.9em; color: #666;'>Compression Ratio</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 2em; font-weight: 700; color: #dc2626;'>1.9%</div>
                <div style='font-size: 0.9em; color: #666;'>Generalization Gap</div>
            </div>
        </div>
    </div>
    """)
    
    with gr.Tab("Tokenizer Demo"):
        gr.Markdown("## ✨ Tokenize Kannada Text")
        gr.Markdown("*Enter Kannada text below and see how it gets tokenized in real-time*")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="📝 Input Kannada Text",
                    placeholder="Type or paste Kannada text here...\nExample: ಕನ್ನಡ ಭಾಷೆ",
                    lines=6,
                    max_lines=10
                )
                
                with gr.Row():
                    tokenize_btn = gr.Button(
                        "🔤 Tokenize", 
                        variant="primary",
                        size="lg"
                    )
                    show_details_checkbox = gr.Checkbox(
                        label="Show detailed breakdown", 
                        value=False,
                        container=False
                    )
                
                gr.Markdown("---")
                gr.Markdown("### 📚 Try These Examples")
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=input_text,
                    label=""
                )
            
            with gr.Column(scale=1):
                output_result = gr.Markdown(
                    label="Results",
                    value="*Tokenization results will appear here*"
                )
                output_details = gr.Markdown(
                    label="",
                    value=""
                )
        
        tokenize_btn.click(
            fn=tokenize_text,
            inputs=[input_text, show_details_checkbox],
            outputs=[output_result, output_details],
            api_name="tokenize"
        )
    
    with gr.Tab("Tokenizer Info"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(get_vocab_info())
                
                gr.Markdown("""
        ## About This Tokenizer
        
        ### Training Details:
        - **Method:** Pure BPE (industry-standard, no morphology preprocessing)
        - **Data Source:** Kannada Wikipedia
        - **Training Samples:** 2,057,673 sentences (373 MB)
        - **Pre-tokenizer:** Whitespace (preserves Kannada word boundaries)
        - **Normalizer:** NFC Unicode (handles combining characters)
        
        ### Performance Metrics:
        - **Compression Ratio:** 4.48 chars/token (50K vocab)
        - **Generalization Gap:** 1.9% (exceptional!)
        - **Fertility:** 1.533 tokens/word (near word-level)
        - **Word Coverage:** 79.6% complete words
        - **Unknown Token Rate:** 0% (perfect coverage)
        
        ### Comparison:
        - Larger than existing Kannada tokenizers (50K vs 32K baseline)
        - Matches GPT-3 vocabulary scale
        - Optimized specifically for Kannada (vs multilingual models)
        
        ### Use Cases:
        - Language modeling (GPT-style models)
        - Machine translation (Kannada ↔ other languages)
        - Text classification (sentiment, topic, etc.)
        - Named entity recognition
        - Question answering systems
        
        ### Technical Specifications:
        - **Library:** HuggingFace Tokenizers
        - **Format:** JSON (portable across frameworks)
        - **Special Tokens:** [PAD], [UNK], [CLS], [SEP], [MASK]
        - **File Size:** ~15 MB
        - **Loading Time:** < 1 second
        
        ### Citation:
        ```
        @misc{kannada-bpe-tokenizer-2025,
          title={Kannada BPE Tokenizer: A 50K-token Byte Pair Encoding Tokenizer for Kannada},
          author={shwethd},
          year={2025},
          note={Trained on Kannada Wikipedia with systematic scaling analysis}
        }
        ```
                """)
    
    with gr.Tab("About & Performance"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
        ## 📊 Performance Highlights
        
        ### Tokenization Quality
        
        **Common Words** - Perfect word-level tokenization:
        - `ಕನ್ನಡ ಭಾಷೆ` → 2 tokens
        - `ಬೆಂಗಳೂರು ನಗರ` → 2 tokens
        
        **Compound Words** - Learned as single units:
        - `ಮಗುವನ್ನು` → 1 token ✅
        - `ಚಳಿಗಾಲ` → 1 token ✅
        
        **Case Markers** - All recognized:
        - `ಮನೆಗೆ`, `ಮನೆಯಿಂದ`, `ಮನೆಯಲ್ಲಿ` → Single tokens each
        
        **Technical Terms** - Complete words:
        - `ಅಂತರರಾಷ್ಟ್ರೀಯ`, `ತಂತ್ರಾಂಶಗಳು`, `ವಿಜ್ಞಾನಿಗಳು` → 1 token each
        
        ---
        
        ## 🔬 Systematic Scaling Study
        
        We tested multiple vocabulary sizes to find the optimal configuration:
        
        | Vocabulary | Compression | Generalization | Status |
        |------------|-------------|----------------|--------|
        | 8,000 | 3.51 | 6.5% gap | Good |
        | 16,000 | 3.73 | - | Better |
        | 32,000 | 4.21 | 6.5% gap | Excellent |
        | **50,000** | **4.48** | **1.9% gap** | **Optimal!** ⭐ |
        | 64,000 | 4.62 | 7.4% gap | Overfitting |
        | 100,000 | 4.81 | 13.1% gap | Too large |
        
        **Finding:** 50K vocabulary achieves the best generalization (1.9% gap) 
        while maintaining excellent compression!
        
        ---
        
        ## 🎯 Quality Metrics
        
        - ✅ **Generalization:** 1.9% gap (excellent real-world performance)
        - ✅ **Coverage:** 0% unknown tokens (perfect)
        - ✅ **Morphology:** 100% consistency
        - ✅ **Word Vocabulary:** 79.6% complete words
        - ✅ **Fertility:** 1.533 tokens/word (near word-level)
        
        ---
        
        ## 🏗️ Technical Details
        
        **Training Data:** 373 MB Kannada Wikipedia (2M sentences)  
        **Algorithm:** Pure BPE (industry-standard)  
        **Pre-tokenizer:** Whitespace (preserves Kannada integrity)  
        **Normalizer:** NFC Unicode (handles combining characters)  
        **Method:** Statistical learning (no linguistic rules)
                """)

# Launch
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Launching Kannada BPE Tokenizer App...")
    print("="*70)
    print(f"\n✓ Tokenizer loaded: {VOCAB_SIZE:,} tokens")
    print("✓ App ready!")
    print("\n🌐 Opening in browser...")
    print("="*70 + "\n")
    
    demo.launch(
        share=True,
        show_error=True,
        quiet=False
    )

