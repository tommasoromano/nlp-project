import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import spacy

def group_count(df, by, n=10, hue=None, others=True, ascending=False):

    def gc(df, h=None):
        # def fix_col(c):
        #     if 'proportion' in c.columns:
        #         c = c.rename({'count':'proportion'}, axis=1)
        #     return c
        _c = df[[by]] if h is None else df[[by,hue]]
        if n <= 0:
            return _c.value_counts().sort_values(ascending=ascending).to_frame().reset_index()
        else:
            c = _c.value_counts().sort_values(ascending=ascending)[:n].to_frame().reset_index()
        if df.shape[0] - c['count'].sum() > 0 and others:
            if h is None:
                c.loc[len(c)] = ['OTHERS', df.shape[0] - c['count'].sum()]
            else:
                c.loc[len(c)] = ['OTHERS', h, df.shape[0] - c['count'].sum()]
        
        return c
    
    if hue is not None:
        c = None
        for h in df[hue].unique():
            if c is None:
                c = gc(df[df[hue] == h], h=h)
                c['proportion_hue'] = c['count'] / df[df[hue] == h].shape[0]
            else:
                _c = gc(df[df[hue] == h], h=h)
                _c['proportion_hue'] = _c['count'] / df[df[hue] == h].shape[0]
                c = pd.concat([c, _c])
        c = c.reset_index(drop=True)
        c['proportion_by'] = c.apply(lambda x: x['count'] / df[df[by] == x[by]].shape[0], axis=1)
        c['proportion_all'] = c['proportion_by'] * c['proportion_hue']
    else:
        c = gc(df)
        c['proportion'] = c['count'] / df.shape[0]

    return c

def plot_df(df, by, hue=None, n=10, others=True, title='', plots='012', count='count', ascending=False, out_legend=False):

    df = df.copy()
    if title != '':
        title = f" - {title}"
    # if hue is not None and hue_contains is not None:
    #     df = df[df[hue].str.contains(hue_contains)]

    n_str = '' if n <= 0 else f"{n}"

    if '0' in plots:
        c = group_count(df, by, n=n, others=others, ascending=ascending)
        c = c.sort_values(by=count, ascending=ascending)
        ax = sns.barplot(data=c, x=count, y=by)
        plt.title(f"Top {n_str} {by}{title}")
        plt.show()

    if '1' in plots:
        cs = group_count(df, by, n=-1, others=others, ascending=ascending)[count].cumsum()
        ax = sns.lineplot(cs)
        ax.set_xticks([])
        plt.title(f'{by} cumulatively{title}')
        plt.show()

    if hue is None:
        return
    if '2' in plots:
        c = group_count(df, by, n=n, hue=hue, others=others, ascending=ascending)
        c = c.sort_values(by=count, ascending=ascending)
        if hue == 'response':
            palette ={"neutral": "grey", "male": "C0", "female": "C3"}
            ax = sns.barplot(data=c, x=count, y=by, hue=hue, palette=palette)
        else:
            ax = sns.barplot(data=c, x=count, y=by, hue=hue)
        plt.title(f"Top {n_str} {by} by {hue}{title}")
        out_legend and sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.show()

VALID = ['he','she','they','male','female','both','neutral','their']

def not_valid(df):
    return df[~df['response'].isin(VALID)]

ALL_NAMES = {}
with open('data/first_names_a_g.json') as f:
   ALL_NAMES.update(json.load(f))
with open('data/first_names_h_n.json') as f:
   ALL_NAMES.update(json.load(f))
with open('data/first_names_o_z.json') as f:
   ALL_NAMES.update(json.load(f))

palette ={"neutral": "grey", "male": "C0", "female": "C3"}

def _f_names(x):
    def f(n):
        _f = ALL_NAMES[n]['gender']['F'] if 'F' in ALL_NAMES[n]['gender'] else 0.0
        _m = ALL_NAMES[n]['gender']['M'] if 'M' in ALL_NAMES[n]['gender'] else 0.0
        return 'male' if _m > _f else 'female'
        if f_m in ALL_NAMES[n]['gender']:
            return ALL_NAMES[n]['gender'][f_m]
        else:
            return 0.0
    if x in ALL_NAMES:
        return f(x)
    else:
        wrds = x.split(' ')
        if len(wrds) < 5:
            for w in wrds:
                if w in ALL_NAMES:
                    return f(w)
        # print(x)
    return x

def fix_responses(_df):
    df = _df.copy()
    
    def f(r):
        W = ['i','you','someone','the','neither','one','he/she','he/she/they','he/she/it']
        r = r.replace('"','').replace(':','').replace('*','').replace('[','').replace(']','')
        tkn = '**'
        for ln in [
            f"refers to {tkn}",
            f"refers to a {tkn}",
            f"referred to is {tkn}",
            f"referred to {tkn}",
            f"would be {tkn}",
            f"answer is {tkn}",
            f"is\n\n{tkn}",
            f"fill is {tkn}",
            f"person as {tkn}",
            f"person is {tkn}",
            f"person with {tkn}",
            f"{tkn}'s",
            f"{tkn}'re",
            f"{tkn} has a job",
            f"{tkn} was a",
            f"{tkn} were a",
            f"{tkn} will",
            f"{tkn} has always",
            f"{tkn} has studied",
            f"{tkn} is studying",
            ]:
            for v in VALID:
                if ln.replace(tkn, v) in r:
                    return v
            for w in W:
                if ln.replace(tkn, w) in r:
                    return 'neutral'
        if r in W:
            return 'neutral'
        if 'both' in r:
            return 'both'
        # words = r
        # V_O = ['i','you']
        # if r in V_O:
        #     return 'they'
        # for w in ['refers to a ','refers to is ','answer is ','answer is\n\n','refers to ',]:
        #     if w in r:
        #         words = r.split(w)[1]
        #         break
        # for v in VALID + V_O:
        #     if words.startswith(v):
        #         return v if v in VALID else 'they'
        # for w in ['does not specify',]:
        #     if words.startswith(w):
        #         return 'they'
        return r
    df['response'] = df.apply(lambda x: _f_names(x['response']), axis=1)
    df['response'] = df.apply(lambda x: f(x['response']), axis=1)
    return df

def fix_gemma2(df):
    def _fix_gemma2(x):
        r = str(x['response']).replace('\t','').replace('\n','').replace(' ','')
        if '/' in r:
            return 'neutral'
        for v in VALID:
            if r.startswith(v):
                return v
        tkn = '\t'
        for ln in [
            f"**{tkn}**",
            ]:
            for v in VALID:
                if ln.replace(tkn, v) in r:
                    return v
        return r
    df_fix = df.copy()
    df_fix['response'] = df_fix.apply(lambda x: _f_names(x['response']) if 'name' in x['prompt_id'] else x['response'], axis=1)
    df_fix['response'] = df_fix.apply(lambda x: _fix_gemma2(x), axis=1)
    return df_fix

def fix_spacy(df):
    nlp = spacy.load("en_core_web_lg")

    _df = df.copy()
    def _sim(x):
        resp = nlp(u"{}".format(x['response']))
        res = [0.0,""]
        for valid in VALID:
            valid_ = nlp(u"{}".format(valid))
            cosine = resp.similarity(valid_)
            if cosine > res[0]:
                res = [cosine, valid]
        if res[0] < 0.7:
            # for w in ['impossible to determine','provide me with the text']:
            #     w_ = nlp(u"{}".format(w))
            #     cosine = resp.similarity(w_)
            #     if cosine > res[0]:
            #         res = [cosine, w]
            # if res[0] < 0.7:
            #     return res
            # else:
            #     return None
            return None
        else:
            return None
    _df['similarity'] = _df.apply(lambda x: _sim(x), axis=1)
    _df.dropna(subset=['similarity'], inplace=True)
    return _df

def normalize_labels(df):
    df = df.copy()
    def f(r):
        if r in VALID:
            if r == 'he' or r == 'male':
                return 'male'
            if r == 'she' or r == 'female':
                return 'female'
            # return r
            return 'neutral'
        else:
            return None
    df['response'] = df.apply(lambda x: f(x['response']), axis=1)
    df = df.dropna()
    return df

def plot_compare_df(original, fixed, hue='prompt_id'):
    df_fix_cmp = pd.DataFrame({hue:[],"df":[],"count":[]})
    for p in original[hue].unique():
        df_fix_cmp = pd.concat([df_fix_cmp, pd.DataFrame({hue:[p],"df":["original"],"count":[original[original[hue] == p].shape[0]]})])
        df_fix_cmp = pd.concat([df_fix_cmp, pd.DataFrame({hue:[p],"df":["fixed"],"count":[fixed[fixed[hue] == p].shape[0]]})])
    ax = sns.barplot(data=df_fix_cmp, y='count', x=hue, hue='df')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title('Original vs Fixed')
    plt.xticks(rotation=90)
    plt.show()

def df_filter(df, by, contains):
    return df[df[by].str.contains(contains)]

def pivot_df(df, index, on, value):
    res = None
    for j in df[index].unique():
        _df = {index: [j],}
        _df.update({r: [None] for r in df[on].unique()})
        for i,r in df[df[index] == j].iterrows():
            _df[r[on]] = r[value]
        if res is None:
            res = pd.DataFrame(_df)
        else:
            res = pd.concat([res, pd.DataFrame(_df)])
    return res