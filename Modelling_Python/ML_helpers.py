from collections import Counter

import pandas as pd
from imblearn.over_sampling import SMOTE

import wget
from os.path import exists

milkGreen = '#0BCB85'

url = "https://pada.psycharchives.org/bitstream/e6d56ab5-8557-4833-8112-9d8e9ed57211"
path_categories = "../data/app_categorisation_2020.csv"
if not exists(path_categories):
    wget.download(url, path_categories)

df_categories = pd.read_csv(path_categories, sep=',')
df_categories.drop(columns=['Perc_users', 'Training_Coding_1', 'Training_Coding_all', 'Training_Coding_2', 'Rater1', 'Rater2'], inplace=True)




def oversampling_smote(df_x_features, df_y_labels):
    print('----oversampling smote-----')
    smote = SMOTE()

    # fit predictor and target variable
    x_smote, y_smote = smote.fit_resample(df_x_features, df_y_labels)

    print('Original dataset shape', Counter(df_y_labels))
    print('Resample dataset shape', Counter(y_smote))
    return x_smote, y_smote





