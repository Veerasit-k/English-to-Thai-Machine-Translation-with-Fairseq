import pandas as pd
import re
import os
import random

def read_txt_to_df_including_null(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    df = pd.DataFrame(lines, columns=['Content'])
    df.index.name = 'Line_Number'
    df.index += 1  # Line numbers start at 1 instead of 0
    return df

def strip_and_calculate_lengths(df, col):
    df[col] = df[col].str.strip()
    df[f'length_{col.split("_")[1]}'] = df[col].str.len()
    return df

def apply_regex_filter(df, col, pattern):
    df[f'Contains_{pattern}'] = df[col].apply(lambda x: bool(re.search(pattern, x)))
    df = df[df[f'Contains_{pattern}'] == False]
    df.drop(columns=[f'Contains_{pattern}'], inplace=True)
    return df

def clean_quote(text):
    replacements = {"&apos;": "'", "apos": "'", "&quot;": '"', "“": '"', "”": '"', "‘": "'", "’": "'"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def clean_url_text(content):
    content = re.sub(r'https?://\S+|www\.\S+', ' ', content)
    content = re.sub(r'\S+@\S+', ' ', content)
    return content

def clean_text_with_exclusions(text, exclusions):
    for char in exclusions:
        text = text.replace(char, '')
    return text.strip()

def replace_contractions(text, contraction_dict, pattern):
    def contraction_replacement(match):
        contraction = match.group().lower()
        expanded_form = contraction_dict[contraction]
        # Handle cases based on context if needed
        if contraction in {"he's", "she's", "it's"}:
            # Context-based replacement: could be "is" or "has"
            if re.search(r"\b(has|been|got|had|having|have)\b", text, re.IGNORECASE):
                return expanded_form.replace(" is", " has")
            else:
                return expanded_form
        return expanded_form

    return pattern.sub(contraction_replacement, text)

def handle_aint(text):
    def replace_aint(match):
        word = match.group(1).lower()
        return {
            "i": "am not",
            "he": "is not", "she": "is not", "it": "is not",
            "there": "is not", "that": "is not",
            "you": "are not", "we": "are not", "they": "are not"
        }.get(word, "is not")

    return re.sub(r"\b(\w+)\s+ain\s?'t\b", lambda m: f"{m.group(1)} {replace_aint(m)}", text, flags=re.IGNORECASE)

def clean_text(text):
    contractions = {
        "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
        "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would",
        "he's": "he is", "he's": "he has", "he'll": "he will", "he'd": "he would",
        "she's": "she is", "she's": "she has", "she'll": "she will", "she'd": "she would",
        "it's": "it is", "it's": "it has", "it'll": "it will", "it'd": "it would",
        "we're": "we are", "we've": "we have", "we'll": "we will", "we'd": "we would",
        "they're": "they are", "they've": "they have", "they'll": "they will", "they'd": "they would",
        "can't": "cannot", "won't": "will not", "don't": "do not", "doesn't": "does not",
        "didn't": "did not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
        "wouldn't": "would not", "shouldn't": "should not", "mightn't": "might not", "mustn't": "must not",
        "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not", "there's": "there is",
        "there're": "there are", "there'll": "there will", "there'd": "there would",
        "here's": "here is", "here're": "here are", "here'll": "here will", "here'd": "here would",
        "where's": "where is", "where're": "where are", "where'll": "where will", "where'd": "where would",
        "what's": "what is", "what're": "what are", "what'll": "what will", "what'd": "what did",
        "who's": "who is", "who're": "who are", "who'll": "who will", "who'd": "who would",
        "that's": "that is", "that're": "that are", "that'll": "that will", "that'd": "that would",
        "let's": "let us", "ain't": "is not", "could've": "could have", "would've": "would have",
        "should've": "should have", "might've": "might have", "must've": "must have", "I'll've": "I will have",
        "you'll've": "you will have", "he'll've": "he will have", "she'll've": "she will have", "we'll've": "we will have",
        "they'll've": "they will have", "I'd've": "I would have", "you'd've": "you would have", "he'd've": "he would have",
        "she'd've": "she would have", "we'd've": "we would have", "they'd've": "they would have", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not", "hasn't": "has not", "haven't": "have not",
        "hadn't": "had not", "doesn't": "does not", "don't": "do not", "didn't": "did not", "won't": "will not",
        "wouldn't": "would not", "shan't": "shall not", "shouldn't": "should not", "can't": "cannot",
        "couldn't": "could not", "mustn't": "must not", "mightn't": "might not", "needn't": "need not",
        "daren't": "dare not", "oughtn't": "ought not", "tisn't": "it is not", "tis": "it is",
        "you'd": "you would", "they'd": "they would", "it'd": "it would", "who'd": "who would",
        "there'd": "there would", "how'd": "how would", "what'd": "what would", "that'd": "that would",
        "this'd": "this would", "those'd": "those would", "these'd": "these would",
        "it'll": "it will", "he'll": "he will", "she'll": "she will", "who'll": "who will",
        "there'll": "there will", "how'll": "how will", "what'll": "what will", "that'll": "that will",
        "this'll": "this will", "those'll": "those will", "these'll": "these will"
    }

    text = handle_aint(text)
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions.keys()) + r')\b', re.IGNORECASE)
    return replace_contractions(text, contractions, pattern)

def clean_spaces(text):
    return re.sub(r'\s+', ' ', text)

def process_text_files(data_name, en_file, th_file):
    print("Step 1: Reading text files")
    en_combined = read_txt_to_df_including_null(en_file)
    th_combined = read_txt_to_df_including_null(th_file)
    merged_df = en_combined.merge(th_combined, left_index=True, right_index=True)

    print("Step 2: Stripping whitespace and calculating lengths")
    merged_df = strip_and_calculate_lengths(merged_df, 'Content_x')
    merged_df = strip_and_calculate_lengths(merged_df, 'Content_y')

    print("Step 3: Filtering based on length")
    merged_df['min_length'] = merged_df[['length_x', 'length_y']].min(axis=1)
    merged_df = merged_df[(merged_df['min_length'] >= 80) & (merged_df['min_length'] <= 300)]

    print("Step 4: Filter out samples with duplicate English content")
    i = merged_df.groupby('Content_x').size().reset_index(name='count_context_x')
    i = i[i['count_context_x'] == 1]
    merged_df = merged_df[merged_df['Content_x'].isin(i['Content_x'])]

    print("Step 5: Dropping duplicate rows")
    merged_df = merged_df.drop_duplicates(subset=['Content_x', 'Content_y'])

    print("Step 6: Applying regex filter to remove English characters in Thai content")
    merged_df = apply_regex_filter(merged_df, 'Content_y', '[a-zA-Z]')

    print("Step 7: Removing incorrectly encoded Thai content")
    incorrect_encoded_pattern = r'เธ[\x80-\xFF]{1,3}|เธ[ˆ]|เธ[“”‰€™]|เน€เธ'
    merged_df = apply_regex_filter(merged_df, 'Content_y', incorrect_encoded_pattern)

    print("Step 8: Cleaning quotes and urls in content")
    merged_df['Content_x'] = merged_df['Content_x'].apply(clean_quote)
    merged_df['Content_y'] = merged_df['Content_y'].apply(clean_quote)
    merged_df['Content_x'] = merged_df['Content_x'].apply(clean_url_text)
    merged_df['Content_y'] = merged_df['Content_y'].apply(clean_url_text)

    print("Step 9: Removing excluded characters")
    en_excluded_chars = ['♪', '#', '&', '!', '©', '☺', '★', '♥', '�', '☆', '☀', '●', '♥', '◄', '☼', '®', '☻', '○', '�', '♦', '□',
                         '฿', '⊕', '⋆', '⇒', '@', '►', '¢', '⊗', '▼', '¬', '░', '⇐', '⊂', '⊃', '▬', '⊗', '$', '*', '\u200b', '\u2024',
                         '▪', '•']
    th_excluded_chars = en_excluded_chars + [',', '.', ':', ';', '?', '_', '|', '`', '~', '^', '—']

    merged_df['Content_x'] = merged_df['Content_x'].apply(lambda x: clean_text_with_exclusions(x, en_excluded_chars).lower())
    merged_df['Content_y'] = merged_df['Content_y'].apply(lambda x: clean_text_with_exclusions(x, th_excluded_chars))
    
    print("Step 10: Cleaning spaces in text")
    merged_df['Content_x'] = merged_df['Content_x'].apply(clean_spaces)
    merged_df['Content_y'] = merged_df['Content_y'].apply(clean_spaces)

    print("Step 11: Removing excluded special characters")
    excluded_chars = ['apos']
    merged_df['Content_x'] = merged_df['Content_x'].apply(lambda x: clean_text_with_exclusions(x, excluded_chars))
    merged_df['Content_y'] = merged_df['Content_y'].apply(lambda x: clean_text_with_exclusions(x, excluded_chars))

    print("Step 12: Cleaning and transforming text")
    merged_df['Content_x'] = merged_df['Content_x'].apply(clean_text)

    print("Step 13: Saving cleaned data to files")
    merged_df['Content_x'].to_csv(f'{data_name}_cleaned_en.txt', index=False, header=False)
    merged_df['Content_y'].to_csv(f'{data_name}_cleaned_th.txt', index=False, header=False)
    print("Process completed successfully.")

data_name = 'hackathon'
process_text_files(data_name=data_name,
                    en_file=r'combined_en.txt',
                    th_file=r'combined_th.txt'
                    )

def split_data(data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    return train_data, valid_data, test_data

# Read tokenized data
with open(f'{data_name}_cleaned_en.txt', 'r', encoding='utf-8') as f:
    en = f.readlines()
with open(f'{data_name}_cleaned_th.txt', 'r', encoding='utf-8') as f:
    th = f.readlines()

# Ensure the English and Thai data have the same length
assert len(en) == len(th)

# Split the data
en_train, en_valid, en_test = split_data(en)
th_train, th_valid, th_test = split_data(th)

# Write the split data to files
with open(f'{data_name}_train.en', 'w', encoding='utf-8') as f:
    f.writelines(en_train)
with open(f'{data_name}_valid.en', 'w', encoding='utf-8') as f:
    f.writelines(en_valid)
with open(f'{data_name}_test.en', 'w', encoding='utf-8') as f:
    f.writelines(en_test)

with open(f'{data_name}_train.th', 'w', encoding='utf-8') as f:
    f.writelines(th_train)
with open(f'{data_name}_valid.th', 'w', encoding='utf-8') as f:
    f.writelines(th_valid)
with open(f'{data_name}_test.th', 'w', encoding='utf-8') as f:
    f.writelines(th_test)