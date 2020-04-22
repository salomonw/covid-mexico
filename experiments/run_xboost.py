import parameters as parameters
import models as models
import data_analysis as DA
import pandas as pd
import os
from datetime import datetime
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def run_xgboost(fname, y, outdir, waiting ):
    # Specify file name


    data = parameters.read_data_eng(fname=fname,
                                    y=y,
                                  percentage_train=0.7,
                                  additional_vars=False,
                                  waiting=waiting)
    [dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS] = data
    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    param['eval_metric'] = ['auc', 'ams@0']

    # fit model no training data
    model = XGBClassifier()
    model.fit(dftrain, y_train)

    y_pred = model.predict(dftest)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    xgb.plot_importance(model)
    plt.tight_layout()
    plt.savefig(outdir+'/f-score.pdf')
    xgb.plot_tree(model, num_trees=10)
    plt.tight_layout()
    plt.savefig(outdir+'/tree.pdf')
    a = xgb.to_graphviz(model, num_trees=2)
    a.render()


# Get timestamp
ts_obj = datetime.now()
ts = ts_obj.strftime("%d-%b-%Y_%H-%M-%S")
os.mkdir('results/'+ts)

out_dir = 'results/' + ts + '/xgboost'
os.mkdir(out_dir)
fname = "data/200416COVID19MEXICO.csv"

for y in ['TIPO_PACIENTE', 'MUERTE', 'UCI', 'INTUBADO']:
    os.mkdir(out_dir+"/"+y)
    os.mkdir(out_dir+"/"+y+"-W")
    run_xgboost(fname=fname, y=y, outdir=out_dir+"/"+y, waiting=False)
    run_xgboost(fname=fname, y=y, outdir=out_dir+"/"+y+"-W", waiting=True)


'''
c = run(y='hosp', fname=fname, name='hosp_Y', out_dir=out_dir, waiting=False, add_vars=['RESULTADO'])
coeff_LC.extend(c)
c = run(y='hosp', fname=fname, name='hosp_Y-W', out_dir=out_dir, waiting=True, add_vars=['RESULTADO'])
coeff_LC.extend(c)

out_dir = 'results/' + ts + '/PredictiveModels/Deaths'
os.mkdir(out_dir)
c = run(y='deaths', fname=fname, name='deaths_Y', out_dir=out_dir, waiting=False, add_vars=['RESULTADO','TIPO_PACIENTE','NEUMONIA', 'INTUBADO', 'UCI'])
coeff_LC.extend(c)
c = run(y='deaths', fname=fname, name='deaths_Y-W', out_dir=out_dir, waiting=True, add_vars=['RESULTADO','TIPO_PACIENTE','NEUMONIA', 'INTUBADO', 'UCI'])
coeff_LC.extend(c)

out_dir = 'results/' + ts + '/PredictiveModels/ICU'
os.mkdir(out_dir)
c = run(y='icu', fname=fname, name='icu_Y', out_dir=out_dir, waiting=False, add_vars=False)
coeff_LC.extend(c)
c = run(y='icu', fname=fname, name='icu_Y-W', out_dir=out_dir, waiting=True, add_vars=False)
coeff_LC.extend(c)

out_dir = 'results/' + ts + '/PredictiveModels/Vent'
os.mkdir(out_dir)
c = run(y='vent', fname=fname, name='vent_Y', out_dir=out_dir, waiting=False, add_vars=False)
coeff_LC.extend(c)
c = run(y='vent', fname=fname, name='vent_Y-W', out_dir=out_dir, waiting=True, add_vars=False)
coeff_LC.extend(c)

'''
