import pandas as pd

def load_labels_from_csv(label_file):
    """
    Loads the 'MEDICINE_NAME' column from the CSV file.
    Returns a list of strings (each string is the medicine name).
    """
    df = pd.read_csv(label_file)
    # The second column is 'MEDICINE_NAME'
    labels = df['MEDICINE_NAME'].tolist()
    return labels

def build_vocabulary(labels):
    """
    Build a character-level vocabulary from a list of strings.
    Adds a blank/pad token (index 0) for CTC.
    """
    unique_chars = set()
    for label in labels:
        for ch in label:
            unique_chars.add(ch)
    # Convert set to list, sort, and enumerate
    sorted_chars = sorted(list(unique_chars))

    # Create a mapping from character to integer
    # index 0 will be used as the 'blank' token for CTC
    char_to_idx = {'<blank>': 0}
    idx = 1
    for ch in sorted_chars:
        char_to_idx[ch] = idx
        idx += 1

    return char_to_idx

def get_vocab_and_mapping(label_file):
    """
    Load labels from CSV and build vocabulary and mapping.
    Returns both char_to_idx and idx_to_char mappings.
    """
    # Load labels
    labels = load_labels_from_csv(label_file)
    
    # Build vocabulary
    char_to_idx = build_vocabulary(labels)
    
    # Create reverse mapping (for decoding)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    
    print(f"Vocabulary size: {len(char_to_idx)}")
    
    return char_to_idx, idx_to_char
