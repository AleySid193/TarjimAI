import streamlit as st
import torch
import pickle
import math
import torch.nn as nn
import torch.nn.functional as F
import os

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = self.word2idx
        self.itos = self.idx2word

    def __len__(self):
        return len(self.word2idx)

    @classmethod
    def tokenize(cls, text):
        return text.split()

    def tokenize_to_ids(self, tokens):
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)
        return self.out(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.norm3(x + self.ff(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        x = self.embedding(src)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.fc_out(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, n_heads=4, d_ff=512, num_layers=2, max_len=5000):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, n_heads, d_ff, num_layers, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, n_heads, d_ff, num_layers, max_len)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)

# Load model and vocabularies with caching
@st.cache_resource
def load_model():
    try:
        # Use relative paths for cloud compatibility
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_checkpoint = os.path.join(current_dir, "transformer_seq2seq_lang_trans.pt")
        src_vocab_file = os.path.join(current_dir, "src_vocab.pkl")
        tgt_vocab_file = os.path.join(current_dir, "tgt_vocab.pkl")

        # Check if files exist
        if not all(os.path.exists(f) for f in [model_checkpoint, src_vocab_file, tgt_vocab_file]):
            st.error("Required model files are missing. Please ensure all files are present in the repository.")
            return None, None, None, None

        with open(src_vocab_file, "rb") as f:
            src_vocab = pickle.load(f)
        with open(tgt_vocab_file, "rb") as f:
            tgt_vocab = pickle.load(f)

        model = TransformerSeq2Seq(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=256,
            n_heads=4,
            d_ff=512,
            num_layers=2
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        model.to(device)
        model.eval()

        return model, src_vocab, tgt_vocab, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def translate(arabic_text, model, src_vocab, tgt_vocab, device):
    try:
        # Tokenize input
        arabic_tokens = arabic_text.split()
        src_ids = [src_vocab.word2idx.get(token, src_vocab.word2idx["<unk>"]) for token in arabic_tokens]
        
        # Generate translation
        model.eval()
        with torch.no_grad():
            src = torch.tensor([src_ids], dtype=torch.long, device=device)
            src_mask = (src != src_vocab.word2idx["<pad>"]).unsqueeze(1).unsqueeze(2)
            enc_out = model.encoder(src, src_mask)

            ys = torch.tensor([[tgt_vocab.word2idx["<sos>"]]], dtype=torch.long, device=device)

            for _ in range(128 - 1):
                tgt_mask = (ys != tgt_vocab.word2idx["<pad>"]).unsqueeze(1).unsqueeze(2)
                seq_len = ys.size(1)
                subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()
                tgt_mask = tgt_mask & ~subsequent_mask

                out = model.decoder(ys, enc_out, src_mask, tgt_mask)
                prob = out[:, -1, :]
                next_word = torch.argmax(prob, dim=1).item()

                ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
                if next_word == tgt_vocab.word2idx["<eos>"]:
                    break

            output_ids = ys[0].cpu().numpy().tolist()
            output_tokens = [tgt_vocab.idx2word[i] for i in output_ids if i not in {tgt_vocab.word2idx["<sos>"], tgt_vocab.word2idx["<eos>"]}]
            
            return " ".join(output_tokens)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(
    page_title="Arabic to English Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stTextArea {
        font-size: 16px;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🌐 Translator Settings")
    st.markdown("---")
    st.markdown("""
    ### About
    This is an Arabic to English translator powered by a Transformer-based neural network.
    
    ### Tips
    - Enter complete sentences
    - Use proper Arabic text formatting
    - Avoid special characters
    - Keep sentences reasonably short
    """)

# Main content
st.title("🌐 TarjimAI")

# Load model
model, src_vocab, tgt_vocab, device = load_model()

if model is None:
    st.error("Failed to load the model. Please check if all required files are present.")
    st.stop()

# Create two columns for input and output
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    arabic_text = st.text_area(
        "Enter Arabic Text",
        placeholder="مرحبا كيف حالك",
        height=150
    )

    if st.button("Translate", type="primary"):
        with st.spinner("Translating..."):
            translation = translate(arabic_text, model, src_vocab, tgt_vocab, device)
            with col2:
                st.subheader("Translation")
                st.text_area("English Translation", translation if translation else "", height=150)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using PyTorch and Streamlit</p>
    </div>
""", unsafe_allow_html=True)
