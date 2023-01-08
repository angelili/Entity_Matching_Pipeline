import pandas as pd
import recordlinkage


data_binning = __import__('Task1.2')
try:
    attrlist = data_binning.__all__
except AttributeError:
    attrlist = dir(data_binning)
for attr in attrlist:
    globals()[attr] = getattr(data_binning, attr)


df_perfect_Match = pd.read_csv('DBLP-ACM_perfectMapping.csv', header=0, encoding="ISO-8859-1")
df_ACM,df_DBLP,candidate_links,candidate_links_all = data_binning.binning()

def matching():
    d_1 = df_ACM['id'].to_dict()
    d_1_flip = {y: x for x, y in d_1.items()}

    d_2 = df_DBLP['id'].to_dict()
    d_2_flip = {y: x for x, y in d_2.items()}

    df_perfect_Match['idACM'] = df_perfect_Match['idACM'].map(d_1_flip)
    df_perfect_Match['idDBLP'] = df_perfect_Match['idDBLP'].map(d_2_flip)

    perfectMapping = df_perfect_Match[['idACM', 'idDBLP']]

    links_true = pd.MultiIndex.from_frame(perfectMapping)

    links_true

    compare_cl = recordlinkage.Compare()
    compare_cl.exact("venue", "venue", label="venue")
    compare_cl.string("title", "title", method="jarowinkler", label="title")
    compare_cl.exact("year", "year", label="year")

    compare_cl.string("authors", "authors", method="jarowinkler", label="authors")
    features = compare_cl.compute(candidate_links, df_ACM, df_DBLP)
    features_all = compare_cl.compute(candidate_links_all, df_ACM, df_DBLP)

    return features,features_all