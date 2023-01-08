from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from operator import ge
import os
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno as msno
import warnings
warnings.filterwarnings('ignore')


os.chdir("Data_Set&Saved _Models")

combined_df = pd.DataFrame
new_df = pd.DataFrame


def get_combined_dataset():
    whitewine_df = pd.read_csv('winequality-white.csv', delimiter=';')
    redwine_df = pd.read_csv('winequality-red.csv', delimiter=';')

    whitewine_df['wine_type'] = 0
    redwine_df['wine_type'] = 1
    combined_df = pd.concat([whitewine_df, redwine_df])
    print(combined_df.describe())
    return combined_df


# Exploring the DataSet

def evaluating_dataset():
    combined_df = get_combined_dataset()
    print(combined_df.info())
    print(combined_df.describe())
    combined_df.isnull().sum()
    from scipy import stats
    z = np.abs(stats.zscore(combined_df))
    print(z)
    msno.bar(combined_df, figsize=(15, 7), color='pink')

    combined_df.hist(bins=25, figsize=(10, 10))
    # display histogram
    plt.show()

    # ploting heatmap (for correlation)
    plt.figure(figsize=[19, 10], facecolor='blue')
    sb.heatmap(combined_df.corr(), annot=True)
    plt.show()

    # distribution of wine type

    fig = px.pie(values=combined_df['wine_type'].value_counts(),
                 names=combined_df['wine_type'].value_counts().index,
                 )
    fig.update_traces(hole=.6, hoverinfo="label+percent", marker=dict(
        colors=['snow', 'tomato'], line=dict(color=['black'], width=1)))
    fig.add_annotation(x=0.50, y=0.5, text='Wine Types',
                       showarrow=False, font=dict(size=20, color='Steelblue'))
    fig.add_annotation(x=0.27, y=0.8, text='Red Wine',
                       showarrow=False, font=dict(size=15, color='tomato'))
    fig.add_annotation(x=0.75, y=0.6, text='White Wine',
                       showarrow=False, font=dict(size=15, color='gold'))

    fig.update_layout(margin={'b': 0, 'l': 0, 'r': 0, 't': 100},
                      paper_bgcolor='rgb(248, 248, 255)',
                      plot_bgcolor='rgb(248, 248, 255)',
                      showlegend=False,
                      title={'font': {
                          'family': 'monospace',
                          'size': 22,
                          'color': 'grey'},
        'text': 'Distribution Of Red & White Wine',
                'x': 0.50, 'y': 1})
    fig.show()

    # Quality Distribution in Red and White wine

    white = combined_df[combined_df['wine_type'] == 0]
    red = combined_df[combined_df['wine_type'] == 1]

    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.35, 0.35],
                        subplot_titles=['White Wine Quality', 'Red Wine Quality'])

    fig.append_trace(go.Bar(x=white['quality'].value_counts().index,
                            y=white['quality'].value_counts(),
                            text=white['quality'].value_counts(),
                            marker=dict(
                            color='snow',
                            line=dict(color='black', width=1)
                            ),
                            name=''
                            ), 1, 1
                     )

    fig.append_trace(go.Bar(x=red['quality'].value_counts().index,
                            y=red['quality'].value_counts(),
                            text=red['quality'].value_counts(),
                            marker=dict(
                            color='coral',
                            line=dict(color='red', width=1)
                            ),
                            name=''
                            ), 1, 2
                     )

    fig.update_traces(textposition='outside')

    fig.update_layout(margin={'b': 0, 'l': 0, 'r': 0, 't': 100},
                      paper_bgcolor='rgb(248, 248, 255)',
                      plot_bgcolor='rgb(248, 248, 255)',
                      showlegend=False,
                      title={'font': {
                          'family': 'monospace',
                          'size': 22,
                          'color': 'grey'},
        'text': 'Quality Distribution In Red & White Wine',
                'x': 0.50, 'y': 1})

    fig.show()

    # Checking for skwness in the data

    plt.figure(figsize=(20, 14))
    for i, col in enumerate(list(combined_df.iloc[:, 1:].columns.values)):
        plt.subplot(4, 3, i+1)
        sns.distplot(combined_df[col], color='r', kde=True, label='data')
        plt.grid()
        plt.legend(loc='upper right')
        plt.tight_layout()

    df1 = combined_df.iloc[:, 1:]
    plot_rows = 4
    plot_cols = 3
    fig = make_subplots(rows=plot_rows, cols=plot_cols)

    x = 0
    for i in range(1, plot_rows + 1):
        for j in range(1, plot_cols + 1):

            fig.add_trace(go.Box(y=df1[df1.columns[x]].values,
                                 name=df1.columns[x],
                                 ),
                          row=i,
                          col=j)

            x = x+1

    fig.update_layout(height=1200, width=1200)

    fig.update_layout(margin={'b': 0, 'l': 0, 'r': 0, 't': 100},
                      paper_bgcolor='rgb(248, 248, 255)',
                      plot_bgcolor='rgb(248, 248, 255)',
                      showlegend=False,
                      title={'font': {
                          'family': 'monospace',
                          'size': 22,
                          'color': 'grey'},
        'text': 'Checking Skewness',
                'x': 0.50, 'y': 1})
    fig.show()


def basic_preprocessing():
    combined_df = get_combined_dataset()

    for a in range(len(combined_df.corr().columns)):
        for b in range(a):
            if abs(combined_df.corr().iloc[a, b]) > 0.7:
                name = combined_df.corr().columns[a]
                print(name)

    # dropping total sulfur dioxide coloumn for low correlation as to reduce features
    new_df = combined_df.drop('total sulfur dioxide', axis=1)

    # In the dataset, there is so much notice data present, which will affect the accuracy of our ML model

    new_df.isnull().sum()

    # We see that there are not many null values are present in our data so we simply fill them with the help of the fillna() function

    new_df.update(new_df.fillna(new_df.mean()))
    scaler = MinMaxScaler(feature_range=(0, 1))

    normal_df = scaler.fit_transform(new_df)
    normal_df = pd.DataFrame(normal_df, columns=new_df.columns)
    print(normal_df.head())

    new_df["good wine"] = ["yes" if i >=
                           7 else "no" for i in new_df['quality']]

    X = normal_df.drop(["quality"], axis=1)
    y = new_df["good wine"]

    y.value_counts()
    sns.countplot(y)
    plt.show()

    return X, y


########################################################################################
#################*******  Regression   *******##########################################
########################################################################################


def split_data_regression(combined_df):
    ds_train, val_test_ds = train_test_split(
        combined_df, test_size=0.2, random_state=1)
    ds_valid, ds_test = train_test_split(
        val_test_ds, test_size=0.5, random_state=1)
    return ds_train, ds_valid, ds_test


def load_data(ds_train, ds_test):
    transformer_cat = make_pipeline(
        # SimpleImputer(strategy="constant", fill_value="NA"), # Fortunately, no missing values
        OneHotEncoder(handle_unknown='ignore'),
    )
    transformer_num = make_pipeline(
        # SimpleImputer(strategy="constant"), # Fortunately, no missing values
        MinMaxScaler(),
    )

    features_num = [
        'fixed acidity', 'volatile acidity', 'citric acid',
        'residual sugar', 'chlorides', 'free sulfur dioxide',
        'total sulfur dioxide', 'density', 'pH', 'sulphates',
        'alcohol'
    ]

    features_cat = ["wine_type"]

    preprocessor = make_column_transformer(
        (transformer_num, features_num),
        (transformer_cat, features_cat),
    )

    X_train = ds_train.drop('quality', axis=1)
    X_train = pd.DataFrame(preprocessor.fit_transform(X_train))
    X_train.columns = features_num + [1, 0]

    y_train = ds_train['quality']

    X_test = ds_test.drop('quality', axis=1)
    X_test = pd.DataFrame(preprocessor.transform(X_test))
    X_test.columns = features_num + [1, 0]

    y_test = ds_test['quality']

    return X_train, X_test, y_train, y_test


def regressionPreprocess():

    # Execute load_data() for training
    df = get_combined_dataset()
    ds_train, ds_valid, ds_test = split_data_regression(df)
    X_train, X_valid, y_train, y_valid = load_data(ds_train, ds_valid)

    # Processed Features (training set - in DataFrame form)
    print(pd.DataFrame(X_train).head())

    return X_train, X_valid, y_train, y_valid


########################################################################################
#################*******  Classification   *******##########################################
########################################################################################


def split_dataset_classification():
    combined_df = get_combined_dataset()

    X = combined_df.iloc[:, 0:11]
    y = np.ravel(combined_df.wine_type)

    # Splitting the data set for training and validating
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=45)

    # Use the same function above for the validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.25, random_state=8)  # 0.25 x 0.8 = 0.2

    return X_train, X_test, y_train, y_test, X_val, y_val


get_combined_dataset()
