import parameters as parameters
import models as models
import data_analysis as DA
import pandas as pd
import os
from datetime import datetime

# Get timestamp
ts_obj = datetime.now()
ts = ts_obj.strftime("%d-%b-%Y_%H-%M-%S")
ts +='_one_example'
os.mkdir('results/'+ts)


def run(y, fname, name, out_dir,  waiting, add_vars=False, max_steps=10000):
    if add_vars != False:
        print(name + " started! (" + str(len(add_vars)) + ")" )
    else:
        print(name + " started! (1)")
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
                                additional_vars=add_vars,
                                waiting=waiting)

    [dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS] = data

    res, coeff_LC = models.run_classification_experiment(dftrain, dftest, y_train, y_test,
                                         CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, out_dir=out_dir, name=name, max_steps=max_steps)

    coeff_LC.rename({'weights': name}, inplace=True)
    coeffs = [coeff_LC]
    if add_vars == False:

        return coeffs

    acc = []
    auc = []
    base = []
    name_ = []
    acc.append(res.accuracy.to_list())
    auc.append(res.auc.to_list())
    base.append(res.accuracy_baseline.to_list()[0])
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
        models.run_classification_experiment(dftrain, dftest, y_train, y_test,
                                            CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,out_dir=out_dir, name=name)
        acc.append(res.accuracy.to_list())
        auc.append(res.auc.to_list())
        base.append(res.accuracy_baseline.to_list()[0])
        name_.append(name)
        if j>=2:
            del vars_[-1]
        j+=1
        print(j)
    res_table=[]
    for i in range(len(acc)):
        l = []
        l.append(name_[i])
        l.extend(acc[i])
        l.append(base[i])
        l.extend(auc[i])
        res_table.append(l)

    coeff_LC.rename({'weights':name}, inplace=True)
    coeffs.append(coeff_LC)
    df = pd.DataFrame(res_table, columns = ['Name',	'Accuracy_LC',	'Accuracy_CART',	'Accuracy_Boosting',	'Accuracy_DNN',	'Accuracy_Baseline',	'AUC_LC',	'AUC_CART',	'AUC_Boosting',	'AUC_DNN'])
    df.to_latex(out_dir + '/results_table.csv', float_format="%.2f")
    print(name + " finished!")
    return coeffs


# Specify file name
fname = "data/200416COVID19MEXICO.csv"


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

## Run predictive models
out_dir = 'results/' + ts + '/PredictiveModels'
os.mkdir(out_dir)

out_dir = 'results/' + ts + '/PredictiveModels/Hospitalizations'
os.mkdir(out_dir)

coeff_LC = []

c = run(y='hosp', fname=fname, name='hosp_Y', out_dir=out_dir, waiting=True, add_vars=False, max_steps=10000)
coeff_LC.extend(c)

i=0
df = coeff_LC[0].T
for coeff in coeff_LC:
    if i>0:
        df = pd.concat([df, coeff.T], axis=1, sort=True)
    i+=1


df.rename(feature_translate, inplace=True)
df.to_latex('results/' + ts + '/weights_LCs.tex', float_format="%.2f")
