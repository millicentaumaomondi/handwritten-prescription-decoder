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
    print(f"Loading models from {model_path}, {ngram_path}, and {label_file}")
    # Load labels and build vocabulary
    vocab, idx_to_char = get_vocab_and_mapping(label_file)
    print("Vocabulary loaded successfully.")

    # Load model
    try:
        model = CRNN_EfficientNet(input_channels=1, hidden_size=256, num_classes=len(vocab)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("CRNN model loaded successfully.")
    except Exception as e:
        print(f"Error loading CRNN model: {str(e)}")
        raise

    # Load GPT-2 model
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        gpt_lm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        gpt_lm.eval()
        print("GPT-2 model loaded successfully.")
    except Exception as e:
        print(f"Error loading GPT-2 model: {str(e)}")
        raise

    # Load n-gram LM
    if os.path.exists(ngram_path):
        try:
            with open(ngram_path, 'rb') as f:
                ngram_lm = pickle.load(f)
            print("N-gram model loaded successfully.")
        except ModuleNotFoundError:
            print("N-gram model not found, creating a new one.")
            df = pd.read_csv(label_file)
            label_corpus = df['MEDICINE_NAME'].tolist()
            ngram_lm = NGramLanguageModel(label_corpus)
            with open(ngram_path, 'wb') as f:
                pickle.dump(ngram_lm, f)
    else:
        print("N-gram model file not found, creating a new one.")
        df = pd.read_csv(label_file)
        label_corpus = df['MEDICINE_NAME'].tolist()
        ngram_lm = NGramLanguageModel(label_corpus)
        with open(ngram_path, 'wb') as f:
            pickle.dump(ngram_lm, f)

    return model, tokenizer, gpt_lm, ngram_lm, vocab, idx_to_char
