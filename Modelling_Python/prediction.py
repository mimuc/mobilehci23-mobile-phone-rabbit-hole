from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from tqdm import tqdm

import ML_helpers
import shap
shap.initjs()
import matplotlib


# setup graphics lib
matplotlib.rcParams["figure.dpi"] = 200
size = 11
ticksize = 11
legendsize = 11
plt.rc('font', size=size)  # controls default text size
plt.rc('axes', titlesize=size)  # fontsize of the title
plt.rc('axes', labelsize=size)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=ticksize)  # fontsize of the x tick labels
plt.rc('ytick', labelsize=ticksize)  # fontsize of the y tick labels
plt.rc('legend', fontsize=legendsize)  # fontsize of the legend
pd.set_option('display.max_columns', None)



def random_forest_classifier(x_train_features, y_train_labels, x_test_features, y_test_labels, filename, report_df, df_sessions):
    feature_list = x_train_features.columns
    print('\n\n***Random Forest***')

    # our final hyperparamters are configured here
    forest = RandomForestClassifier(
        max_features="sqrt",
        n_estimators=10,
        max_depth=8,
        criterion="gini",
        random_state=42,
        min_samples_leaf=1,
        min_samples_split=10,
        n_jobs=10
    )
    forest.fit(x_train_features, y_train_labels)
    print("number of features: "+str(len(feature_list)))
    print(forest.classes_)

    print("----- Report Train (Run 1) -----")
    y_predict = forest.predict(x_train_features)
    score = metrics.accuracy_score(y_train_labels, y_predict)
    print(score)
    print(metrics.classification_report(y_train_labels, y_predict))  # output_dict=True))

    print("----- Report Validation (Run 2) -----")
    y_predict = forest.predict(x_test_features)
    print(metrics.classification_report(y_test_labels, y_predict))  # output_dict=True))
    report = pd.DataFrame.from_dict(metrics.classification_report(y_test_labels, y_predict, output_dict=True))
    report['target'] = filename
    report['algorithm'] = "random_forest"

    print("-----score----------")
    score = metrics.accuracy_score(y_test_labels, y_predict)
    print(score)
    report['score'] = score

    print("---------confusion matrix--------")
    cm = confusion_matrix(y_test_labels, y_predict)
    print(cm)

    print("-------importance--------")
    importance = pd.DataFrame({'feature': feature_list, 'importance': np.round(forest.feature_importances_, 3)})
    importance.sort_values('importance', ascending=False, inplace=True)
    print(importance)
    importance.to_csv(fr'./{filename}-randomForest_f_importance.csv')

    print("---------------SHAP PLOTS-------------")
    print(f'0: {forest.classes_[0]}, 1: {forest.classes_[1]}')
    explainer = shap.TreeExplainer(forest)
    shap_values = explainer.shap_values(x_train_features, check_additivity=False, approximate=True)
    print(type(x_train_features))


    plt.figure()
    fig = plt.figure(dpi=200)
    shap.summary_plot(shap_values[0], x_train_features, plot_type="bar", max_display=20, show=False)
    plt.title("Random Forest - Shap values 0")
    fig.tight_layout()
    matplotlib.rcParams["figure.dpi"] = 200
    plt.show()
    # plt.savefig("plt1.png")
    #
    fig = plt.figure(dpi=200)
    shap.summary_plot(shap_values[1], x_train_features, plot_type="dot", max_display=20, show=False)
    plt.title("Random Forest - Shap values 1")
    fig.tight_layout()
    plt.show()
    fig.savefig("shap-beaswarm-1-top20.pdf")


    index_of_clickfreq = x_train_features.columns.get_loc("f_click_frequency")
    s1 = pd.Series(shap_values[1][:, index_of_clickfreq])
    s2 = x_train_features['f_click_frequency']
    df = pd.DataFrame({'feature': s1, 'shapvalue': s2})
    fig = plt.scatter(x=s1, y=s2, c=s2, cmap="coolwarm", s=3)
    plt.show()

    fig2 = shap.dependence_plot("f_click_frequency", shap_values[1], x_train_features, interaction_index="f_click_frequency")
   #fig2.savefig("shap-scatter-clickfreq.pdf")


    return pd.concat([report_df, report]), forest




report_all = pd.DataFrame()


path = r'../data/smartphone_sessions_with_features.pickle'

# import sessions and keep only those which are labelled
df_sessions = pd.read_pickle(path)
df_sessions = df_sessions[df_sessions['target_label'] != 'NaN']
df_sessions = df_sessions[~df_sessions['target_label'].isnull()]

# cleanup
df_sessions = df_sessions.drop(['session_length'],axis=1) # f_session_length exists instead (numeric)
df_sessions = df_sessions.fillna(0)

lstKeep = []
for c in df_sessions.columns:

    if ("f_hour_of_day_" in c):
        lstKeep.append(c)

timeValues = [(lambda x: float(x.split("_")[-1]))(x) for x in lstKeep]
df_sessions["f_hour_of_day"] = df_sessions[lstKeep].apply(lambda x: sum(x.values * timeValues), axis=1)

lstKeep = []
for c in df_sessions.columns:

    if ("f_weekday_" in c):
        lstKeep.append(c)

timeValues = [(lambda x: float(x.split("_")[-1]))(x) for x in lstKeep]
df_sessions["f_weekday"] = df_sessions[lstKeep].apply(lambda x: sum(x.values * timeValues), axis=1)

df_sessions['f_click_frequency'] = df_sessions['f_clicks'] / df_sessions['f_session_length']
df_sessions['f_scroll_frequency'] = df_sessions['f_scrolls'] / df_sessions['f_session_length']

# remove users, go get equal set as group prediction:
df_sessions = df_sessions[df_sessions.studyID != 'AN09BI']
df_sessions = df_sessions[df_sessions.studyID != 'NI23HE']
df_sessions = df_sessions[df_sessions.studyID != 'PI02MA']


# list of features that should be kept for modelling
lstFeatures = [
    'f_click_frequency',
    'f_clicks',
    'f_count_session_1h',
    'f_count_session_2h',
    'f_count_session_3h',
    'f_glances_since_last_session',
    'f_hour_of_day',
    'f_internet_connected_WIFI',
    'f_internet_connected_mobile',
    'f_internet_disconnected',
    'f_internet_else',
    'f_ringer_mode_normal',
    'f_ringer_mode_silent',
    'f_ringer_mode_unknown',
    'f_ringer_mode_vibrate',
    'f_scroll_frequency',
    'f_scrolls',
    'f_time_since_last_session',
    'f_weekday'
]

for c in df_sessions.columns:
    if ("f_weekday_" in c):
        lstFeatures.append(c)

for c in df_sessions.columns:
    if ("f_hour_of_day_" in c):
        lstFeatures.append(c)

for c in df_sessions.columns:
    if ("_category_" in c):
        None
        lstFeatures.append(c)





lstFeatures = sorted(lstFeatures)
len(sorted(lstFeatures))

# divide into train, test and validate datasets
ids = df_sessions['studyID'].unique()
trainIDs = ids[:15]
validationIDs = ids[15:18]
testIDs = ids[18:]
print(trainIDs, validationIDs, testIDs)
print(len(trainIDs), len(validationIDs), len(testIDs))

dfTrain = df_sessions[df_sessions.studyID.isin(trainIDs)]
dfVal = df_sessions[df_sessions.studyID.isin(validationIDs)]
dfTest = df_sessions[df_sessions.studyID.isin(testIDs)]

xTrain, yTrain = ML_helpers.oversampling_smote(dfTrain[lstFeatures], dfTrain["target_label"])
xTest, yTest  = ML_helpers.oversampling_smote(dfTest[lstFeatures], dfTest["target_label"])
xVal, yVal = ML_helpers.oversampling_smote(dfVal[lstFeatures], dfVal["target_label"])



def rf_gridsearch():
    print('Grid Search starting.')
    param_grid = {
        'n_estimators': [5, 10, 100, 200, 500, 700],  # , 1000], #, 2000, 3000],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [4, 5, 6, 7, 8, 10, 12, 14, None],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
    }
    lst = []
    for e in param_grid['n_estimators']:
        for mf in param_grid['max_features']:
            for md in param_grid['max_depth']:
                for c in param_grid['criterion']:
                    for minl in param_grid['min_samples_leaf']:
                        for mins in param_grid['min_samples_split']:
                            for runs in range(5):
                                lst.append([e, mf, md, c, minl, mins])
    len(lst)


    def doJob(x):
        e, mf, md, c, minl, mins = x
        starttime = time.time()
        forest = RandomForestClassifier(
            max_features=mf,
            n_estimators=e,
            max_depth=md,
            criterion=c,
            min_samples_leaf=minl,
            min_samples_split=mins,
            n_jobs=1
        )
        forest.fit(xTrain, yTrain)
        endtime = time.time()
        y_predict = forest.predict(xTrain)
        scoreTrain = metrics.accuracy_score(yTrain, y_predict)

        y_predict = forest.predict(xVal)
        scoreTest = metrics.accuracy_score(yVal, y_predict)

        return ["RF", e, mf, md, c, minl, mins, scoreTrain, scoreTest, endtime - starttime]


    with Pool(40) as p:
        r = list(tqdm(p.imap(doJob, lst), total=len(lst)))


    dfStats = pd.DataFrame(r)
    dfStats.columns = ["Model", "n_estimators", "max_features", "max_depth", "criterion", "min_samples_leaf", "min_samples_split", "AccTrain", "AccTest", "TrainTime"]
    dfStats = dfStats.groupby(["Model", "n_estimators", "max_features", "max_depth", "min_samples_leaf", "min_samples_split", "criterion"]).describe()
    dfStats = dfStats[[("AccTrain", "mean"), ("AccTrain", "std"), ("AccTest", "mean"), ("AccTest", "std"), ("TrainTime", "mean")]]
    dfStats.columns = ["AccTrainMean", "AccTrainSD", "AccTestMean", "AccTestSD", "TrainTimeMean"]
    dfStats = dfStats.reset_index()
    dfStats = dfStats.sort_values("AccTestMean", ascending=False)
    dfStats[dfStats.AccTestMean >.875]
    dfStats.to_pickle('dfStatsGridSearch.pickle')


    print('Grid Search done.')


### Grid Search ###
# If you want to run the grid search, uncomment this:
#rf_gridsearch()



print('ML start.')
filename = "atleastone_more_than_intention"


report_all, forest = random_forest_classifier(xTrain, yTrain, xVal, yVal, filename, report_all,df_sessions)

print("----- Report Validation (run 3) -----")
y_predict = forest.predict(xTest)
print(metrics.classification_report(yTest, y_predict))  # output_dict=True))
print(metrics.accuracy_score(yTest, y_predict))

print('ML done.')

