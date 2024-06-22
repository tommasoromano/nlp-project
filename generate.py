import ollama
import pandas as pd
import os
from transformers import pipeline
from transformers import DistilBertTokenizer
import nlp_synt_data as sd
from nlp_synt_data import *
from constants import *

# ollama run llama3:instruct
# ollama run mistral:instruct
# ollama run gemma:instruct
# ollama run phi3:instruct

# !wget https://github.com/nikhgarg/EmbeddingDynamicStereotypes/raw/master/data/occupation_percentages_gender_occ1950.csv
# !wget https://github.com/nikhgarg/EmbeddingDynamicStereotypes/raw/master/data/occupation_percentages_race_occ1950.csv
# !wget https://github.com/nikhgarg/EmbeddingDynamicStereotypes/raw/master/data/female_pairs.txt
# !wget https://github.com/nikhgarg/EmbeddingDynamicStereotypes/raw/master/data/male_pairs.txt
# !wget https://github.com/philipperemy/name-dataset/raw/master/names_dataset/v3/first_names.zip
# !wget https://github.com/philipperemy/name-dataset/raw/master/names_dataset/v3/last_names.zip
# !wget https://github.com/MilaNLProc/honest/raw/main/resources/binary/en_template.tsv


models = [
  'llama3:instruct'
  ]

if __name__ == '__main__':


  # Generate synthetic data and run models

  prompt_dict = Utils.list_to_dict(pd.read_csv('data/prompts.csv').values.tolist())
  prompts = PromptGenerator.generate(prompt_dict, [
                                      ['mask-zsl'],
                                      ['label-zsl']
                                      ])
  
  texts = pd.read_csv('data/templates.csv').values.tolist()

  if False:
  
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = pipeline('fill-mask', model='distilbert-base-uncased')
    bert_data = DataGenerator.generate([(t[0].replace('[PERSON]',tokenizer.mask_token),t[1]) for t in TEMPLATES], {
      'JOB': [j for j in JOBS],
      'LABEL': [l for l in LABELS],
      })
    
    def bert_model(prompt, text):
      res = model(f"{prompt}. {text}")
      tot = sum([l['score'] for l in res])
      return str([(l['token_str'],round(l['score']/tot,4)) for l in res])

    ResponseGenerator.generate("bert_results.csv", bert_data, [("none","none")],
          bert_model,
          save_every=100
    )
  
  else:

    data = DataGenerator.generate(TEMPLATES, {
      'JOB': [j for j in JOBS],
      'LABEL': [l for l in LABELS],
      })

    ResponseGenerator.generate("llama_results.csv", data, prompts,
                                lambda prompt, text: ollama.chat(model='llama3:instruct', messages=[
          { 'role': 'system', 'content': prompt, },
          { 'role': 'user', 'content': text, },
          ])['message']['content'],
          save_every=10
    )
