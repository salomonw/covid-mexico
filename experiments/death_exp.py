import os
import sys
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
import math
import experiments.parameters as parameters


dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS = parameters.read_data_death_exp()


feature_columns = []



def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


def make_input_fn(data_df, label_df, num_epochs=500, shuffle=True, batch_size=100):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dftest, y_test,  num_epochs=1, batch_size=len(y_test), shuffle=False)


from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

def classify(feature_columns, train_input_fn, eval_input_fn, method='LC'):
    if method == 'LC':
        est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    elif method == 'DT':
        est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=1)
    elif method == 'DNN':
        est = tf.estimator.DNNClassifier([20, 400, 400,400, 5], feature_columns)

    est.train(train_input_fn)
    result = est.evaluate(eval_input_fn)
    pred_dicts = list(est.predict(eval_input_fn))
    probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
    fpr, tpr, _ = roc_curve(y_test, probs)
    del est
    return result, fpr, tpr




df_result = pd.DataFrame(columns=['method', 'accuracy', 'accuracy_baseline', 'auc', 'auc_precision_recall',
       'average_loss', 'label/mean', 'loss', 'precision', 'prediction/mean',
       'recall', 'global_step'])



for method in ['LC', 'DT', 'DNN']:
    result, fpr, tpr = classify(feature_columns, train_input_fn, eval_input_fn, method=method)
    result['method'] = method
    plt.plot(fpr, tpr, label=method)
    df0 = pd.DataFrame([result], columns=result.keys())
    df_result = pd.concat([df_result, df0], axis=0)#.reset_index()
    del df0


plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.plot([0, 1], [0, 1], color='k')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend(loc=4)
plt.show()
