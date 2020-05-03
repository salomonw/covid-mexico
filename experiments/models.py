from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
import xgboost as xgb
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def one_hot_cat_column(feature_name, vocab):
    """
    Create one hot encoding for a tf classification task

    Parameters
    ----------

    feature_name: feature of interest
    vocab: the list of categories in the feature

    Returns
    -------
    a tf column object

    """
    return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))


def make_input_fn(data_df, label_df, num_epochs=300, shuffle=True, batch_size=100):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function


def get_coefficients_LC(est, feature_columns):
    coeff_ftrl = {}
    coeff = {}
    for name in est.get_variable_names():
        if "Ftrl" in name and 'accumulator' not in name:
            for feature in feature_columns:
                if feature[0][0] in name:
                    if feature[0][0] not in 'EDAD':
                        coeff_ftrl[feature[0][0]] = {}
                        idx = feature[0][1].index(1)
                        coeff_ftrl[feature[0][0]]['weights'] = est.get_variable_value(name)[idx][0]
                    else:
                        coeff_ftrl[feature[0][0]] = {}
                        coeff_ftrl[feature[0][0]]['weights'] = est.get_variable_value(name)[0]
        if "Ftrl" not in name:
            for feature in feature_columns:
                if feature[0][0] in name:
                    if feature[0][0] not in 'EDAD':
                        coeff[feature[0][0]] = {}
                        idx = feature[0][1].index(1)
                        coeff[feature[0][0]]['weights'] = est.get_variable_value(name)[idx][0]
                    else:
                        coeff['EDAD'] = {}
                        coeff['EDAD']['weights'] = est.get_variable_value(name)[0][0]
    return coeff, coeff_ftrl



def classify(feature_columns, train_input_fn, eval_input_fn, y_test, method='LC', max_steps=10000):
    if method == 'LC':
        est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    elif method == 'CART':
        est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                                   n_batches_per_layer=1,
                                                  n_trees=1, max_depth=20, learning_rate=0.1)
    elif method == 'DNN':
        est = tf.estimator.DNNClassifier([20, 400, 400, 100], feature_columns)
    elif method == 'Boosting':
        est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                                  n_batches_per_layer=100,
                                                  n_trees=1000, max_depth=4, learning_rate=0.1)
    est.train(train_input_fn)#, max_steps=max_steps)

    # get coefficients of LC
    if method == 'LC':
        coeff, coeff_ftrl = get_coefficients_LC(est, feature_columns)
        coeff_ftrl_dict = pd.DataFrame.from_dict(coeff_ftrl)
        coeffs_dict = pd.DataFrame.from_dict(coeff)
    else:
        coeff_ftrl_dict = {}
        coeffs_dict = {}

    result = est.evaluate(eval_input_fn)
    pred_dicts = list(est.predict(eval_input_fn))
    probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

    del est
    return result, y_test, probs, coeffs_dict #fpr, tpr, coeffs_dict


def classify_xgboost(dftrain, dftest, y_train,
                    y_test, name='exp', out_dir=''):
    feature_translate = {
        'SEXO': 'Gender',
        'EMBARAZO': 'Pregnant',
        'DIABETES': 'Diabetes',
        'EPOC': 'COPD',
        'ASMA': 'Asthma',
        'INMUSUPR': 'Immunosuppression',
        'HIPERTENSION': 'Hypertension',
        'OTRA_COM': 'Other',
        'CARDIOVASCULAR': 'Cardiovascular Disease',
        'OBESIDAD': 'Obesity ',
        'RENAL_CRONICA': 'Chronic Renal Insufficiency',
        'TABAQUISMO': 'Tobacco Use',
        'OTRO_CASO': 'Contact COVID',
        'NEUMONIA': 'Pneumonia',
        'INTUBADO': 'Ventilator',
        'UCI': 'ICU',
        'HABLA_LENGUA_INDIG': 'Indigenous Len',
        'TIPO_PACIENTE': 'Hospitalization',
        'MUERTE': 'Death',
        'RESULTADO': 'Test',
        'EDAD': 'Age',
        'COPD': 'Age'
    }

    col_d = {i:i.split('_1')[0] for i in list(dftrain)}

    dftrainT = dftrain.rename(columns=col_d)
    dftestT = dftest.rename(columns=col_d)

    dftrainT = dftrainT.rename(columns=feature_translate)
    dftestT = dftestT.rename(columns=feature_translate)
    y_trainT = y_train.rename(columns=feature_translate)
    y_testT = y_test.rename(columns=feature_translate)

    model = XGBClassifier(silent=False,
                          scale_pos_weight=1,
                          learning_rate=0.1,
                          colsample_bytree = 0.4,
                          subsample = 0.8,
                          objective='binary:logistic',
                          n_estimators=1000,
                          reg_alpha = 0.3,
                          max_depth=4,
                          gamma=10)

    model.fit(dftrainT, y_trainT)
    y_pred = model.predict_proba(dftestT)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred[1]

    xgb.plot_importance(model)

    plt.title('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(out_dir+'/'+ name+ '_f-score.pdf')
    plt.savefig(out_dir + '/' + name + '_f-score.png')
    plt.close()

    coeff =  model.feature_importances_
    coeffs = {}
    i=0
    for col in list(dftrain):
        coeffs[col] = [coeff[i]]
        i += 1
    coeffs = pd.DataFrame.from_dict(coeffs)

    return y_test, y_pred, coeffs


def get_roc_curve(y_test, probs):
    fpr, tpr, _ = roc_curve(y_test, probs)
    return fpr, tpr

def get_metrics(y_test, probs):
    result = {}
    predictions = [round(value) for value in probs]
    result['Accuracy'] = metrics.accuracy_score(y_test, predictions)
    result['F1w'] = metrics.f1_score(y_test, predictions, average='weighted')
    result['AUC'] = metrics.roc_auc_score(y_test, probs)
  #  result['Precision'] = metrics.precision_score(y_test, predictions)
    return result


def find_correleted_var(dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS):
    df = pd.concat([dftrain, y_train], axis=1)
    dftest = pd.concat([dftest, y_test], axis=1)
    cols = list(dftrain)
    df = df[cols]
    dftest = dftest[cols]
    catcols = cols.copy()

    for i in NUMERIC_COLUMNS:
        catcols.remove(i)

    for col in catcols:
        dummies = pd.get_dummies(df[col], drop_first='True', prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)
        dummiestest = pd.get_dummies(dftest[col], drop_first='True', prefix=col)
        dftest = pd.concat([dftest, dummiestest], axis=1)
        dftest.drop(col, axis=1, inplace=True)

    correlated_features = set()
    correlated_matrix = df.corr()

    for i in range(len(correlated_matrix.columns)):
        for j in range(i):
            if abs(correlated_matrix.iloc[i, j]) > 0.8:
                colname = correlated_matrix.columns[i]
                correlated_features.add(colname)
    return df, dftest,  correlated_features




def feature_elimination(dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, out_dir, name):
    #dftrainC, dftestC, corr_vars = find_correleted_var(dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS,
    #                                                   NUMERIC_COLUMNS)
    dftrainC, dftestC,  correlated_features = find_correleted_var(dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS)
    #print(correlated_features)
    #for i in correlated_features:
    #    dftrainC.drop(i, axis=1, inplace=True)
    #    dftestC.drop(i, axis=1, inplace=True)


    #'''
    for i in list(dftrainC):  
        if '97' in i:
            dftrainC.drop(i, axis=1, inplace=True)
        elif '98' in i:
            dftrainC.drop(i, axis=1, inplace=True)
        elif '99' in i:
            dftrainC.drop(i, axis=1, inplace=True)
    for i in list(dftestC):
        if '97' in i:
            dftestC.drop(i, axis=1, inplace=True)
        elif '98' in i:
            dftestC.drop(i, axis=1, inplace=True)
        elif '99' in i:
            dftestC.drop(i, axis=1, inplace=True)
    #'''
    categ = list(dftrainC)
    for i in NUMERIC_COLUMNS:
        categ.remove(i)

    X = dftrainC
    y1 = y_train

    rfc = LogisticRegression(penalty='l1', solver='saga')

    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='roc_auc')
    rfecv.fit(X,y1)

    plt.figure(figsize=(4,4))
    #plt.title('Recursive FE with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('AUC', fontsize=14, labelpad=20)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
    plt.xlim(len(X.T), 1)
    plt.grid(True)
    plt.tight_layout()
    #print(out_dir + name)
    plt.savefig(out_dir + "/" + str(name) + '_FE.pdf')
    plt.savefig(out_dir + "/" + str(name) + '_FE.png')
    return dftrainC, dftestC, categ


#from sklearn import metrics, cross_validation
#scores = .cross_val_score(logreg, X, y, cv=10)
from sklearn.calibration import CalibratedClassifierCV

def classify_sklearn(dftrain, dftest, y_train, y_test,
                     CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
                     method):
    if method=='skl-SVM-l1':
        #clf = svm.SVC(probability=True, kernel='linear')#, max_iter=10000)
        clf1 = LinearSVC(penalty='l1',dual=False, max_iter=10000)
        clf = CalibratedClassifierCV(clf1, cv=StratifiedKFold(10))

        clf.fit(dftrain, y_train)

        coef_avg = 0
        for i in clf.calibrated_classifiers_:
            coef_avg = coef_avg + i.base_estimator.coef_
        coeff = coef_avg / len(clf.calibrated_classifiers_)

        i = 0
        coeffs = {}
        for col in list(dftrain):
            coeffs[col] = [coeff[0][i]]
            i += 1
        coeffs = pd.DataFrame.from_dict(coeffs)


    if method == 'skl-LR-l1':
        clf = linear_model.LogisticRegressionCV(#cv=5,#StratifiedKFold(10),
                                                penalty='l1',
                                                dual=False,
                                                solver='saga',
                                                max_iter=10000)
        clf = linear_model.LogisticRegression(penalty='l1',
                                                dual=False,
                                                solver='saga',
                                                max_iter=10000)
        clf.fit(dftrain, y_train)
        coeff = clf.coef_
        i = 0
        coeffs = {}
        for col in list(dftrain):
            coeffs[col] = [coeff[0][i]]
            i += 1
        coeffs = pd.DataFrame.from_dict(coeffs)


    if method == 'skl-RF':
        clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=4)
        clf.fit(dftrain, y_train)
        coeffs = {}


    y_pred = [i[1] for i in clf.predict_proba(dftest)]

    return y_test, y_pred, coeffs





def run_classification_experiment(dftrain, dftest, y_train,
                                  y_test, CATEGORICAL_COLUMNS,
                                  NUMERIC_COLUMNS, name='exp', out_dir='', max_steps=10000):
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    train_input_fn = make_input_fn(dftrain, y_train)
    eval_input_fn = make_input_fn(dftest, y_test,  num_epochs=1, batch_size=len(y_test), shuffle=False)


    plt.figure(figsize=(4,4))

    df_result = {}
    coeffs_df = pd.DataFrame()

    for method in ['skl-SVM-l1', 'skl-LR-l1', 'skl-RF', 'xgboost']:#, 'LC', 'Boosting']:#, 'CART', 'DNN', 'skl-SVM-l1']:
        if method == 'xgboost':
            y_test, probs, coeff = classify_xgboost(dftrain, dftest, y_train,
                                                 y_test, name=name, out_dir=out_dir)
        elif 'skl' in method:
            y_test, probs, coeff = classify_sklearn(dftrain=dftrain,
                                                    dftest=dftest,
                                                    y_train=y_train,
                                                    y_test=y_test,
                                                    CATEGORICAL_COLUMNS=CATEGORICAL_COLUMNS,
                                                    NUMERIC_COLUMNS=NUMERIC_COLUMNS,
                                                    method=method)#, name='exp', out_dir='')
        else:
            result, y_test, probs, coeff = classify(feature_columns, train_input_fn, eval_input_fn, y_test,
                                                    method=method, max_steps=max_steps)

        fpr, tpr = get_roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=method.replace('skl-',''))

        if method == 'LC':
            coeff = pd.DataFrame.from_dict(coeff).T
            coeff_LC = coeff
            coeff = coeff.rename(columns={'weights':'LC'})
            coeffs_df = pd.concat([coeffs_df, coeff], axis=1, sort=True)

        elif method == 'skl-LR-l1':
            coeff = pd.DataFrame.from_dict(coeff).T
            coeff = coeff.rename(columns={0: 'LR-l1'})
            coeffs_df = pd.concat([coeffs_df, coeff], axis=1, sort=True)

        elif method == 'skl-SVM-l1':
            coeff = pd.DataFrame.from_dict(coeff).T
            coeff = coeff.rename(columns={0: 'SVM-l1'})
            coeffs_df = pd.concat([coeffs_df, coeff], axis=1, sort=True)

        elif method == 'xgboost':
            coeff = pd.DataFrame.from_dict(coeff).T
            coeff = coeff.rename(columns={0: 'xgboost'})
            coeffs_df = pd.concat([coeffs_df, coeff], axis=1, sort=True)




        df_result[method] = get_metrics(y_test, probs)

    coeffs_df = coeffs_df.round(3)
    coeffs_df = coeffs_df.sort_values(by=['LR-l1'], ascending = False)

    coeffs_df = coeffs_df.T
    col_d = {i:i.split("_1")[0] for i in list(coeffs_df)}
    feature_translate = {
        'SEXO': 'Gender',
        'EMBARAZO': 'Pregnant',
        'DIABETES': 'Diabetes',
        'EPOC': 'COPD',
        'ASMA': 'Asthma',
        'INMUSUPR': 'Immunosuppression',
        'HIPERTENSION': 'Hypertension',
        'OTRA_COM': 'Other',
        'CARDIOVASCULAR': 'Cardiovascular Disease',
        'OBESIDAD': 'Obesity ',
        'RENAL_CRONICA': 'Chronic Renal Insufficiency',
        'TABAQUISMO': 'Tobacco Use',
        'OTRO_CASO': 'Contact COVID',
        'NEUMONIA': 'Pneumonia',
        'INTUBADO': 'Ventilator',
        'UCI': 'ICU',
        'HABLA_LENGUA_INDIG': 'Indigenous Len',
        'TIPO_PACIENTE': 'Hospitalization',
        'MUERTE': 'Death',
        'RESULTADO': 'Test',
        'EDAD': 'Age',
        'COPD': 'Age'
    }
    coeffs_df = coeffs_df.rename(columns=col_d)
    coeffs_df = coeffs_df.rename(columns=feature_translate)

    coeffs_df = coeffs_df.T

    coeffs_df['LR-l1'] = coeffs_df['LR-l1'] / (coeffs_df['LR-l1'].abs().max())
    coeffs_df['SVM-l1'] = coeffs_df['SVM-l1'] / (coeffs_df['SVM-l1'].abs().max())
    #coeffs_df['xgboost'] = coeffs_df['xgboost'] / (coeffs_df['xgboost'].abs().max())
    coeffs_df = coeffs_df.round(3)
    coeffs_df.to_latex(out_dir + "/" + str(name) + '_coeffs.tex')
    coeffs_df.to_csv(out_dir + "/" + str(name) + '_coeffs.csv')

    df_result = pd.DataFrame(df_result)
    df_result = df_result.round(3)
    print(df_result)

    #plt.title('ROC curve (' + str(name) + ',' + str(len(y_train)) +')')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.plot([0, 1], [0, 1], color='k')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig(out_dir + "/" + str(name) + '.png')
    plt.savefig(out_dir +"/"+ str(name)+'.pdf')

    df_result.to_latex(out_dir +"/"+ str(name) + '.tex')
    df_result.to_csv(out_dir + "/" + str(name) + '.csv')
    plt.close('all')
    coeff_LC= {}
    #coeff_LC.to_latex(out_dir +"/"+ str(name) + '_coeffs_LC.tex')
    return  df_result.T, coeff_LC



def run_recursive_feature_elimination(dftrain, ytrain, y, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, name='exp', out_dir=''):
    df = pd.concat([dftrain, ytrain], axis=1)
    cols = CATEGORICAL_COLUMNS.copy()
    cols.extend(NUMERIC_COLUMNS)
    df = df[cols]
    
    for col in CATEGORICAL_COLUMNS:
        dummies = pd.get_dummies(df[col], drop_first='True', prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)

    correlated_features = set()
    correlated_matrix = df.corr()
    
    for i in range(len(correlated_matrix.columns)):
        for j in range(i):
            if abs(correlated_matrix.iloc[i,j]) >0.8:
                colname = correlated_matrix.columns[i]
                correlated_features.add(colname)

    for i in list(correlated_features):
        df.drop(i, axis=1, inplace=True)

    X = df
    y1 = ytrain

    rfc = LogisticRegression(penalty='l1', solver='saga')

    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='roc_auc')
    rfecv.fit(X,y1)

    plt.figure(figsize=(16, 9))
    plt.title('Recursive FE with Cross-Validation (' + y + ')', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
    plt.grid(True)
    print(out_dir + name)
    plt.savefig(out_dir + name + '.pdf')
    plt.savefig(out_dir + name + '.png')
    return correlated_features