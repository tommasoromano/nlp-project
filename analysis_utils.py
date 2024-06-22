import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def group_count(df, by='response', n=10, hue=None, others=True):

    def gc(df, h=None):
        _c = df[[by]] if h is None else df[[by,hue]]
        if n <= 0:
            return _c.value_counts().sort_values(ascending=False).to_frame().reset_index()
        else:
            c = _c.value_counts().sort_values(ascending=False)[:n].to_frame().reset_index()
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
            else:
                _c = gc(df[df[hue] == h], h=h)
                c = pd.concat([c, _c])
        c = c.reset_index(drop=True)
    else:
        c = gc(df)

    return c

def plot_df(df, by='response', n=10, hue='prompt_id', others=True):

    df = df.copy()
    # if hue is not None and hue_contains is not None:
    #     df = df[df[hue].str.contains(hue_contains)]

    n_str = '' if n <= 0 else f"{n}"

    c = group_count(df, by=by, n=n, others=others)
    sns.barplot(data=c, x='count', y=by)
    plt.title(f"Top {n_str} {by}")
    plt.show()

    cs = group_count(df, by=by, n=-1, others=others)['count'].cumsum()
    ax = sns.lineplot(cs)
    ax.set_xticks([])
    plt.title(f'{by} cumulatively')
    plt.show()

    if hue is None:
        return
    c = group_count(df, by=by, n=n, hue=hue, others=others)
    sns.barplot(data=c, x='count', y=by, hue=hue)
    plt.title(f"Top {n_str} {by} by {hue}")
    plt.show()

VALID = ['he','she','they','male','female','both','neutral']

def not_valid(df):
    return df[~df['response'].isin(VALID)]

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
    df['response'] = df.apply(lambda x: f(x['response']), axis=1)
    return df

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

def  plot_compare_df(original, fixed, hue='prompt_id'):
    df_fix_cmp = pd.DataFrame({hue:[],"df":[],"count":[]})
    for p in original[hue].unique():
        df_fix_cmp = pd.concat([df_fix_cmp, pd.DataFrame({hue:[p],"df":["original"],"count":[original[original[hue] == p].shape[0]]})])
        df_fix_cmp = pd.concat([df_fix_cmp, pd.DataFrame({hue:[p],"df":["fixed"],"count":[fixed[fixed[hue] == p].shape[0]]})])
    ax = sns.barplot(data=df_fix_cmp, y='count', x=hue, hue='df')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title('Original vs Fixed')
    plt.show()