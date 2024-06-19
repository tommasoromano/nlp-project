import ollama
import pandas as pd
import os
import nlp_synt_data as sd
from nlp_synt_data import *

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

  jobs = pd.read_csv('data/occupation_percentages_gender_occ1950.csv')
  jobs = jobs[jobs['Census year'] == 2015]
  jobs['label'] = 'neutral'
  jobs = jobs[['Occupation','label']].values.tolist()

  LABELS_ETHNICITY = ['white','hispanic','asian','black','african']
  LABELS_RELIGION = ['buddhist','christian','hinds','jewish','muslim']
  LABELS_SEXUALITY = ['omosexual','straight']
  LABELS_POLITICAL = ['conservative','liberal']
  data = DataGenerator.generate(texts, {
    'JOB': [j for j in jobs],
    'LABEL': []
    })
  
  ResponseGenerator.generate("results.csv", data, prompts,
                               lambda prompt, text: ollama.chat(model='llama3:instruct', messages=[
        { 'role': 'system', 'content': prompt, },
        { 'role': 'user', 'content': text, },
        ])['message']['content']
        # lambda prompt, text: "response"
  )
