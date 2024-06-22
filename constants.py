import pandas as pd

DF_TEMPLATES = pd.read_csv('data/templates.csv')
TEMPLATES = DF_TEMPLATES.values.tolist()
# TEMPLATES = TEMPLATES + [t.replace('[JOB]', '[LABEL] [JOB]') for t in TEMPLATES]

DF_STATS_GENDER = pd.read_csv('data/occupation_percentages_gender_occ1950.csv')
DF_STATS_GENDER = DF_STATS_GENDER[DF_STATS_GENDER['Census year'] == 2015]

DF_STATS_LABEL = pd.read_csv('data/occupation_percentages_race_occ1950.csv')
DF_STATS_LABEL = DF_STATS_LABEL[DF_STATS_LABEL['Census year'] == 2015]

LABELS = pd.read_csv('data/labels.csv').values.tolist()

JOBS = [(j, 'neutral') for j in DF_STATS_GENDER['Occupation'].values.tolist()]