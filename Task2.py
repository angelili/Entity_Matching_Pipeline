from sklearn.model_selection import train_test_split
import recordlinkage
import recordlinkage as rl
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



data_matching = __import__('Task1.3')
try:
    attrlist = data_matching.__all__
except AttributeError:
    attrlist = dir(data_matching)
for attr in attrlist:
    globals()[attr] = getattr(data_matching, attr)

df_ACM,df_DBLP,features,features_all,links_true = data_matching.matching()

def split_dataset_classification():
    compare_cl = recordlinkage.Compare()
    X_train, X_test = train_test_split(features_all, test_size=0.2)
    Y_train, Y_test = train_test_split(links_true, test_size=0.2)

    features_test_all = compare_cl.compute(Y_test, df_ACM, df_DBLP)
    features_test_all

    return X_train,X_test,Y_train,Y_test

def LogisticRegressionClassifier():
    logreg = rl.LogisticRegressionClassifier()
    X_train, X_test, Y_train, Y_test = split_dataset_classification()
    golden_pairs = features
    golden_matches_index = golden_pairs.index.intersection(links_true)
    print(golden_matches_index)

    logreg.fit(X_train, Y_train)
    print("Intercept: ", logreg.intercept)
    print("Coefficients: ", logreg.coefficients)

    result_logreg = logreg.predict(features_all)

    print(result_logreg)

    rl.confusion_matrix(links_true, result_logreg, len(features_all))

    rl.fscore(links_true, result_logreg)

def svm():
    svm = rl.SVMClassifier()
    X_train, X_test, Y_train, Y_test = split_dataset_classification()
    golden_pairs = features
    golden_matches_index = golden_pairs.index.intersection(links_true)

    svm.fit(X_train, Y_train)
    result_svm = svm.predict(features_all)

    rl.confusion_matrix(links_true, result_svm, len(features_all))

    rl.fscore(links_true, result_svm)


# Since SVM have better results

def hyperparametertuning():
    #To-do
