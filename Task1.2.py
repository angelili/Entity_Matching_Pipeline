import Levenshtein
import numpy as np
import recordlinkage
from recordlinkage.index import Block
from recordlinkage.index import Full




data_prep = __import__('Task1.1')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)

df_DBLP,df_ACM = data_prep.preprocessing

v_1=df_ACM['venue'].value_counts()
v_2=df_DBLP['venue'].value_counts()

v_1_index=v_1.index
v_2_index=v_2.index

def similarities():
    lista = []
    lista_2 = []
    maksimumi = np.zeros((len(v_1_index), len(v_2_index)))
    for i in range(len(v_1_index)):
        for j in range(len(v_2_index)):
            maksimumi[i][j] = Levenshtein.ratio(v_1_index[i], v_2_index[j])
            print(maksimumi)
    from numpy import unravel_index
    for i in range(len(v_1_index)):
        (u, v) = unravel_index(maksimumi.argmax(), maksimumi.shape)
        lista.append(v_1_index[u])
        lista_2.append(v_2_index[v])
        maksimumi[:, v] = np.zeros(len(v_1_index))
        maksimumi[u, :] = np.zeros(len(v_2_index))
        print(maksimumi)
    print(lista)
    print(lista_2)

    dictionary = dict(zip(lista, lista_2))
    dictionary

    df_ACM['venue'] = df_ACM['venue'].map(dictionary)
    y_1 = df_ACM['year'].value_counts()
    y_2 = df_DBLP['year'].value_counts()
    result = y_2.gt(y_1)
    result



def binning():
    indexer = recordlinkage.Index()
    indexer.block('year')

    candidate_links = indexer.index(df_ACM, df_DBLP)

    indexer = Block(left_on=['year', 'venue'],
                    right_on=['year', 'venue'])
    candidate_links = indexer.index(df_ACM, df_DBLP)

    indexer = recordlinkage.Index()
    indexer.add(Full())
    candidate_links_all = indexer.index(df_ACM, df_DBLP)

    return df_ACM,df_DBLP,candidate_links,candidate_links_all








