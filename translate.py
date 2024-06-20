import pandas as pd
from fairseq.models.transformer import TransformerModel
from concurrent.futures import ThreadPoolExecutor

# Load the model
en2th = TransformerModel.from_pretrained(
  'checkpoints',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path='data-bin/translation/',
  bpe='subword_nmt',
  bpe_codes='bpe_th.codes'
)

# Function to translate a single sentence
def translate_sentence(sentence):
    return en2th.translate(sentence)

# Load the test dataset
test_df = pd.read_csv('test.csv')

# Use ThreadPoolExecutor to apply the translation in parallel
with ThreadPoolExecutor(max_workers=20) as executor:
    test_df['translation'] = list(executor.map(translate_sentence, test_df['source']))

# Drop the source column
test_df.drop(columns=['source'], inplace=True)

# Save the translated sentences to a new CSV file
test_df.to_csv('submission.csv', index=False)

