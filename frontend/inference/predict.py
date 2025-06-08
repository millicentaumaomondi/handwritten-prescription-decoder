import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import math
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- NGram Language Model ---
class NGramLanguageModel:
    def __init__(self, corpus, n=5):
        self.n = n
        self.ngrams = Counter()
        self.contexts = Counter()
        self.vocab = set()
        for sentence in corpus:
            tokens = ['<s>'] + list(sentence) + ['</s>']
            for token in tokens:
                self.vocab.add(token)
            for i in range(len(tokens)-n+1):
                self.ngrams[tuple(tokens[i:i+n])] += 1
                self.contexts[tuple(tokens[i:i+n-1])] += 1

    def score(self, sequence):
        tokens = ['<s>'] + list(sequence) + ['</s>']
        log_prob = 0.0
        for i in range(len(tokens)-self.n+1):
            ngram = tuple(tokens[i:i+self.n])
            context = tuple(tokens[i:i+self.n-1])
            prob = (self.ngrams.get(ngram, 0) + 1) / (self.contexts.get(context, 0) + len(self.vocab))
            log_prob += math.log(prob)
        return log_prob

# --- Image preprocessing ---
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = image.convert("L")
    return transform(image).unsqueeze(0)

# --- Beam Search Decoding with optional LM ---
def beam_search_with_ctc_and_lm(
    logits, idx_to_char, beam_width=10, blank_index=0,
    ngram_lm=None, gpt_lm=None, tokenizer=None, lm_weight=0.4, vocab=None
):
    T, B, C = logits.shape
    probs = F.softmax(logits, dim=2)
    results = []

    for b in range(B):
        beams = [([], 0.0, None)]
        for t in range(T):
            new_beams = []
            topk_probs, topk_indices = torch.topk(probs[t, b], beam_width * 2)
            for seq, score, prev_idx in beams:
                for i in range(topk_probs.size(0)):
                    idx = topk_indices[i].item()
                    prob = topk_probs[i].item()
                    if idx == blank_index or idx == prev_idx:
                        new_beams.append((seq.copy(), score + math.log(prob + 1e-10), prev_idx))
                        continue
                    new_seq = seq.copy()
                    new_seq.append(idx)
                    new_beams.append((new_seq, score + math.log(prob + 1e-10), idx))
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            beams = new_beams

        rescored_beams = []
        for seq, ctc_score, _ in beams:
            decoded_seq = ''.join([idx_to_char[i] for i in seq if i in idx_to_char])
            lm_score = 0.0
            if ngram_lm:
                try:
                    lm_score += ngram_lm.score(decoded_seq)
                except:
                    pass
            if gpt_lm and tokenizer and decoded_seq.strip() != "":
                try:
                    inputs = tokenizer(decoded_seq, return_tensors="pt").to(gpt_lm.device)
                    outputs = gpt_lm(**inputs, labels=inputs.input_ids)
                    gpt_loss = outputs.loss.item()
                    lm_score += -gpt_loss
                except:
                    pass
            final_score = (1 - lm_weight) * ctc_score + lm_weight * lm_score
            rescored_beams.append((decoded_seq, final_score))

        best_seq = sorted(rescored_beams, key=lambda x: x[1], reverse=True)[0][0]
        results.append(best_seq)

    return results

# --- Inference wrapper ---
def predict_image(image: Image.Image, model, idx_to_char, vocab=None, ngram_lm=None, gpt_lm=None, tokenizer=None):
    image_tensor = preprocess_image(image).to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(image_tensor)
    prediction = beam_search_with_ctc_and_lm(
        logits,
        idx_to_char=idx_to_char,
        vocab=vocab,
        ngram_lm=ngram_lm,
        gpt_lm=gpt_lm,
        tokenizer=tokenizer,
        lm_weight=0.4
    )
    return prediction[0]
