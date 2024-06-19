import pandas as pd

DF_TEMPLATES = pd.read_csv('data/templates.csv')
TEMPLATES = DF_TEMPLATES.values.tolist()

DF_STATS_GENDER = pd.read_csv('occupation_percentages_gender_occ1950.csv')
DF_STATS_GENDER = DF_STATS_GENDER[DF_STATS_GENDER['Census year'] == 2015]

DF_STATS_LABEL = pd.read_csv('data/occupation_percentages_race_occ1950.csv')
DF_STATS_LABEL = DF_STATS_LABEL[DF_STATS_LABEL['Census year'] == 2015]

LABELS = """white,ethnicity
hispanic,ethnicity
asian,ethnicity
black,ethnicity
african,ethnicity
buddhist,religion
christian,religion
hinds,religion
jewish,religion
muslim,religion
omosexual,sexuality
straight,sexuality
conservative,political
liberal,political
"""
LABELS = [row.split(',') for row in LABELS.split('\n')]

JOBS = [(j, 'neutral') for j in DF_STATS_GENDER['Occupation'].values.tolist()]