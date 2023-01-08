from sklearn.model_selection import train_test_split



data_matching = __import__('Task1.3')
try:
    attrlist = data_matching.__all__
except AttributeError:
    attrlist = dir(data_matching)
for attr in attrlist:
    globals()[attr] = getattr(data_matching, attr)


def split_dataset_classification():
    X_train, X_test = train_test_split(features_all, test_size=0.2)
    Y_train, Y_test = train_test_split(links_true, test_size=0.2)

    features_test_all = compare_cl.compute(Y_test, df_ACM, df_DBLP)
    features_test_all

    