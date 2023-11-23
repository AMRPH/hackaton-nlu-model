from transformers import T5ForConditionalGeneration, T5Tokenizer

import pandas as pd

MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
#model.cuda();
#model.eval();

def paraphrase(text, beams=5, grams=4, do_sample=False):
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams, num_beams=beams, max_length=max_size, do_sample=do_sample)
    return tokenizer.decode(out[0], skip_special_tokens=True)

dataset = pd.read_csv('../train_dataset_train.csv', delimiter=';')

new = dataset.iloc[:5]

for i, msg in enumerate(dataset.iloc[:5, 2]):
    text = paraphrase(msg)
    new.iloc[i, 2] = text
    print(text)

dataset = dataset._append(new, ignore_index = True)

dataset.to_csv('augment.csv', encoding='utf-8')