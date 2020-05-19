import parameters as parameters
import models as models

import pandas as pd
import os
from datetime import datetime


def run_data_analysis(fname):
    ## Run all data analysis figures
    print('Data Analytics started ...')
    os.mkdir('results/' + ts + '/DataAnalysis/')
    DA.run(fname=fname, out_dir='results/' + ts + '/DataAnalysis')
    print('Data Analytics ready!')

def run(y, fname, name, out_dir,  waiting, add_vars=False, max_steps=10000):
    if add_vars != False:
        print(y + ' ' + name + " started! (" + str(len(add_vars)) + ")" )
    else:
        print(y + ' ' + name + " started! (1)")
    if y=='deaths':
        y_spanish = 'MUERTE'
    elif y=='icu':
        y_spanish = 'UCI'
    elif y=='vent':
        y_spanish = 'INTUBADO'
    elif y=='hosp':
        y_spanish = 'TIPO_PACIENTE'

    data = parameters.read_data(fname=fname, y=y_spanish,
                                percentage_train=0.7,
                                additional_vars=False,
                                waiting=waiting)

    [dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS] = data


    ## Feature elimination and dataset structure
    dftrain, dftest, CATEGORICAL_COLUMNS = models.feature_elimination(dftrain, dftest, y_train, y_test , CATEGORICAL_COLUMNS, NUMERIC_COLUMNS)

    res, coeff_LC = models.run_classification_experiment(dftrain, dftest, y_train, y_test,
                                         CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, out_dir=out_dir, name=name, max_steps=max_steps)

    coeff_LC.rename({'weights': name}, inplace=True)
    coeffs = [coeff_LC]
    print(1)
    if add_vars == False:
        return coeffs

    acc = []
    auc = []
    #precision = []
    F1w = []
    name_ = []
    acc.append(res.Accuracy.to_list())
    auc.append(res.AUC.to_list())
    #precision.append(res.Precision.to_list())
    F1w.append(res.F1w.to_list())
    name_.append(name)

    vars_=[]
    j=0
    for var in add_vars:
        vars_.append(var)
        name += '_'
        name += var
        data = parameters.read_data(fname=fname, y=y_spanish,
                                      percentage_train=0.7,
                                      additional_vars=vars_,
                                      waiting=waiting)

        [dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS] = data

        dftrain, dftest, CATEGORICAL_COLUMNS = models.feature_elimination(dftrain, dftest, y_train, y_test,
                                                                          CATEGORICAL_COLUMNS, NUMERIC_COLUMNS)

        res, coeff_LC = models.run_classification_experiment(dftrain, dftest, y_train, y_test,
                                                            CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
                                                             out_dir=out_dir, name=name, max_steps=max_steps)
        acc.append(res.Accuracy.to_list())
        auc.append(res.AUC.to_list())
        #precision.append(res.Precision.to_list())
        F1w.append(res.F1w.to_list())
        name_.append(name)

        if j>=2:
            del vars_[-1]
        j+=1
        print(j+1)
    res_table=[]
    for i in range(len(acc)):
        l = []
        l.append(acc[i])
        l.append(name_[i])
        l.append(auc[i])
        l.append(F1w[i])
        res_table.append(l)

    coeff_LC.rename({'weights':name}, inplace=True)
    coeffs.append(coeff_LC)
    df = pd.DataFrame(res_table, columns = ['Accuracy',	'Name',	'AUC',	'F1w'])
    df.to_latex(out_dir + '/'+ name + '_results_table.tex', float_format="%.2f")
    print(name + " finished!")
    return coeffs

def run_model(y, fname, name, out_dir,  waiting, hosp=False,  add_vars=False, max_steps=10000):
    if add_vars != False:
        print(y + ' ' + name + " started! (" + str(len(add_vars)) + ")" )
    else:
        print(y + ' ' + name + " started! (1)")
    if y=='deaths':
        y_spanish = 'MUERTE'
    elif y=='icu':
        y_spanish = 'UCI'
    elif y=='vent':
        y_spanish = 'INTUBADO'
    elif y=='hosp':
        y_spanish = 'TIPO_PACIENTE'
    elif y=='pneu':
        y_spanish = 'NEUMONIA'

    data = parameters.read_data(fname=fname, y=y_spanish,
                                percentage_train=0.70,
                                additional_vars=add_vars,
                                waiting=waiting,
                                hosp=hosp)

    [dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS] = data


    ## Feature elimination and dataset structure
    dftrain, dftest, CATEGORICAL_COLUMNS = models.feature_elimination(dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, out_dir=out_dir, name=name)



    res, coeff_LC = models.run_classification_experiment(dftrain, dftest, y_train, y_test,
                                         CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, out_dir=out_dir, name=name, max_steps=max_steps)

    #coeff_LC.rename({'weights': name}, inplace=True)
    coeffs = [coeff_LC]
    return coeffs

def run_exp(fname, feature_translate):
    ## Run predictive models
    out_dir = 'results/' + ts + '/PredictiveModels'
    os.mkdir(out_dir)

    coeff_LC = []

    out_dir = 'results/' + ts + '/PredictiveModels/Hospitalizations'
    os.mkdir(out_dir)
    c = run(y='hosp', fname=fname, name='Y', out_dir=out_dir, waiting=False, add_vars=['RESULTADO'])
    coeff_LC.extend(c)
    c = run(y='hosp', fname=fname, name='Y-W', out_dir=out_dir, waiting=True, add_vars=['RESULTADO'])
    coeff_LC.extend(c)

    out_dir = 'results/' + ts + '/PredictiveModels/Deaths'
    os.mkdir(out_dir)
    c = run(y='deaths', fname=fname, name='Y', out_dir=out_dir, waiting=False,
            add_vars=['RESULTADO', 'TIPO_PACIENTE', 'NEUMONIA', 'INTUBADO', 'UCI'])
    coeff_LC.extend(c)
    c = run(y='deaths', fname=fname, name='Y-W', out_dir=out_dir, waiting=True,
            add_vars=['RESULTADO', 'TIPO_PACIENTE', 'NEUMONIA', 'INTUBADO', 'UCI'])
    coeff_LC.extend(c)

    out_dir = 'results/' + ts + '/PredictiveModels/ICU'
    os.mkdir(out_dir)
    c = run(y='icu', fname=fname, name='Y', out_dir=out_dir, waiting=False, add_vars=False)
    coeff_LC.extend(c)
    c = run(y='icu', fname=fname, name='Y-W', out_dir=out_dir, waiting=True, add_vars=False)
    coeff_LC.extend(c)

    out_dir = 'results/' + ts + '/PredictiveModels/Vent'
    os.mkdir(out_dir)
    c = run(y='vent', fname=fname, name='Y', out_dir=out_dir, waiting=False, add_vars=False)
    coeff_LC.extend(c)
    c = run(y='vent', fname=fname, name='Y-W', out_dir=out_dir, waiting=True, add_vars=False)
    coeff_LC.extend(c)

    i = 0
    df = coeff_LC[0].T
    for coeff in coeff_LC:
        if i > 0:
            df = pd.concat([df, coeff.T], axis=1, sort=True)
        i += 1

    df.rename(feature_translate, inplace=True)
    df.to_latex('results/' + ts + '/weights_LCs.tex', float_format="%.2f")
    ## Run predictive models
    out_dir = 'results/' + ts + '/PredictiveModels'
    os.mkdir(out_dir)

    coeff_LC = []

    out_dir = 'results/' + ts + '/PredictiveModels/Hospitalizations'
    os.mkdir(out_dir)


    c = run(y='hosp', fname=fname, name='Y-W', out_dir=out_dir, waiting=True, add_vars=['RESULTADO'])
    coeff_LC.extend(c)

    out_dir = 'results/' + ts + '/PredictiveModels/Deaths'
    os.mkdir(out_dir)
    c = run(y='deaths', fname=fname, name='Y', out_dir=out_dir, waiting=False,
            add_vars=['RESULTADO', 'TIPO_PACIENTE', 'NEUMONIA', 'INTUBADO', 'UCI'])
    coeff_LC.extend(c)
    c = run(y='deaths', fname=fname, name='Y-W', out_dir=out_dir, waiting=True,
            add_vars=['RESULTADO', 'TIPO_PACIENTE', 'NEUMONIA', 'INTUBADO', 'UCI'])
    coeff_LC.extend(c)

    out_dir = 'results/' + ts + '/PredictiveModels/ICU'
    os.mkdir(out_dir)
    c = run(y='icu', fname=fname, name='Y', out_dir=out_dir, waiting=False, add_vars=False)
    coeff_LC.extend(c)
    c = run(y='icu', fname=fname, name='Y-W', out_dir=out_dir, waiting=True, add_vars=False)
    coeff_LC.extend(c)

    out_dir = 'results/' + ts + '/PredictiveModels/Vent'
    os.mkdir(out_dir)
    c = run(y='vent', fname=fname, name='Y', out_dir=out_dir, waiting=False, add_vars=False)
    coeff_LC.extend(c)
    c = run(y='vent', fname=fname, name='Y-W', out_dir=out_dir, waiting=True, add_vars=False)
    coeff_LC.extend(c)

def run_exp_2(fname):
    ## Run predictive models
    out_dir = 'results/' + ts + '/PredictiveModels'
    os.mkdir(out_dir)

    coeff_LC = []

    out_dir = 'results/' + ts + '/PredictiveModels/Hospitalizations'
    os.mkdir(out_dir)
    c = run_model(y='hosp', fname=fname, name='Y-W-0', out_dir=out_dir, waiting=False, hosp=False, add_vars=['RESULTADO'])
    coeff_LC.extend(c)

    out_dir = 'results/' + ts + '/PredictiveModels/Deaths'
    os.mkdir(out_dir)
    c = run_model(y='deaths', fname=fname, name='Y-W-0', out_dir=out_dir, waiting=True, hosp=True, add_vars=['RESULTADO'])
    coeff_LC.extend(c)
    c = run_model(y='deaths', fname=fname, name='Y-W-1', out_dir=out_dir, waiting=True, hosp=True, add_vars=['RESULTADO', 'NEUMONIA', 'INTUBADO', 'UCI'])
    coeff_LC.extend(c)

    out_dir = 'results/' + ts + '/PredictiveModels/ICU'
    os.mkdir(out_dir)
    c = run_model(y='icu', fname=fname, name='Y-W-0', out_dir=out_dir, waiting=True, hosp=True, add_vars=['RESULTADO'])
    coeff_LC.extend(c)
    c = run_model(y='icu', fname=fname, name='Y-W-1', out_dir=out_dir, waiting=True, hosp=True, add_vars=['RESULTADO', 'NEUMONIA'])
    coeff_LC.extend(c)

    out_dir = 'results/' + ts + '/PredictiveModels/Vent'
    os.mkdir(out_dir)
    c = run_model(y='vent', fname=fname, name='Y-W-0', out_dir=out_dir, waiting=True, hosp=True, add_vars=['RESULTADO'])
    coeff_LC.extend(c)
    c = run_model(y='vent', fname=fname, name='Y-W-1', out_dir=out_dir, waiting=True, hosp=True, add_vars=['RESULTADO', 'NEUMONIA', 'UCI'])
    coeff_LC.extend(c)


# Get timestamp
ts_obj = datetime.now()
ts = ts_obj.strftime("%d-%b-%Y_%H-%M-%S")
os.mkdir('results/'+ts)

import data_analysis as DA
import glob
import os
# Specify file name
fname = "data/200506COVID19MEXICO.csv"
fname = max(glob.glob("data//*.csv") , key = os.path.getctime)

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
run_data_analysis(fname)
#run_exp(fname, feature_translate)
run_exp_2(fname)









#TODO: add number of data points
#TODO: add cross-validation for the trained model.
#TODO: add a website that uses predictor online

