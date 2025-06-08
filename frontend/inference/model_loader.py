import torch
import pickle
import pandas as pd
import os
import sys
from .predict import NGramLanguageModel
from .crnn_model import CRNN_EfficientNet
from .vocab import get_vocab_and_mapping
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_all_models(model_path, ngram_path, label_file):
    # Load labels and build vocabulary
    vocab, idx_to_char = get_vocab_and_mapping(label_file)

    # Load model
    model = CRNN_EfficientNet(input_channels=1, hidden_size=256, num_classes=len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load GPT-2 model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt_lm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    gpt_lm.eval()

    # Load n-gram LM
    if os.path.exists(ngram_path):
        try:
            with open(ngram_path, 'rb') as f:
                ngram_lm = pickle.load(f)
        except ModuleNotFoundError:
            # If pickle fails due to module import, create a new n-gram model
            df = pd.read_csv(label_file)
            label_corpus = df['MEDICINE_NAME'].tolist()
            ngram_lm = NGramLanguageModel(label_corpus)
            with open(ngram_path, 'wb') as f:
                pickle.dump(ngram_lm, f)
    else:
        # Load labels for n-gram model
        df = pd.read_csv(label_file)
        label_corpus = df['MEDICINE_NAME'].tolist()
        ngram_lm = NGramLanguageModel(label_corpus)
        with open(ngram_path, 'wb') as f:
            pickle.dump(ngram_lm, f)

    return model, tokenizer, gpt_lm, ngram_lm, vocab, idx_to_char
