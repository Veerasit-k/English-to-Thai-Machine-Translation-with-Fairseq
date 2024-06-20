# %%
import pandas as pd
from fairseq.models.transformer import TransformerModel
en2th = TransformerModel.from_pretrained(
  'checkpoints',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path='data-bin/translation/',
  bpe='subword_nmt',
  bpe_codes='bpe_th.codes'
)

english_sentences = [
    "The concert will start at 8 PM.",
    "The wedding ceremony was beautiful.",
    "I am attending a meeting today.",
    "The festival is happening next week.",
    "I have a doctor's appointment tomorrow.",
    "He is recovering from surgery.",
    "She needs to take her medicine regularly.",
    "Regular exercise is important for health.",
    "I am planning a trip to Japan.",
    "We booked a flight to New York.",
    "He visited the famous landmarks in Paris.",
    "She is traveling for business.",
    "Call 911 in case of an emergency.",
    "There was a fire in the building.",
    "He had to go to the emergency room.",
    "The ambulance arrived quickly.",
    "She graduated with honors.",
    "The school year starts in September.",
    "He is studying for his exams.",
    "The teacher assigned homework."
]

thai_sentences = [en2th.translate(sentence) for sentence in english_sentences]

df = pd.DataFrame({
    "English Sentences": english_sentences,
    "Thai Translations": thai_sentences
})

print(df)
