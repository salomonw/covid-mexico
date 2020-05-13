import parameters as parameters
import models as models
import glob
import os
from datetime import datetime

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

    data = parameters.read_data_age_groups(fname=fname, y=y_spanish,
                                percentage_train=0.7,
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
    c = run_model(y='deaths', fname=fname, name='Y-W-0', out_dir=out_dir, waiting=False, hosp=True, add_vars=['RESULTADO'])
    coeff_LC.extend(c)
    c = run_model(y='deaths', fname=fname, name='Y-W-1', out_dir=out_dir, waiting=False, hosp=True, add_vars=['RESULTADO', 'NEUMONIA', 'INTUBADO', 'UCI'])
    coeff_LC.extend(c)

    out_dir = 'results/' + ts + '/PredictiveModels/ICU'
    os.mkdir(out_dir)
    c = run_model(y='icu', fname=fname, name='Y-W-0', out_dir=out_dir, waiting=False, hosp=True, add_vars=['RESULTADO'])
    coeff_LC.extend(c)
    c = run_model(y='icu', fname=fname, name='Y-W-1', out_dir=out_dir, waiting=False, hosp=True, add_vars=['RESULTADO', 'NEUMONIA'])
    coeff_LC.extend(c)

    out_dir = 'results/' + ts + '/PredictiveModels/Vent'
    os.mkdir(out_dir)
    c = run_model(y='vent', fname=fname, name='Y-W-0', out_dir=out_dir, waiting=False, hosp=True, add_vars=['RESULTADO'])
    coeff_LC.extend(c)
    c = run_model(y='vent', fname=fname, name='Y-W-1', out_dir=out_dir, waiting=False, hosp=True, add_vars=['RESULTADO', 'NEUMONIA', 'UCI'])
    coeff_LC.extend(c)


# Get timestamp
ts_obj = datetime.now()
ts = ts_obj.strftime("%d-%b-%Y_%H-%M-%S")
os.mkdir('results/'+ts)


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

run_exp_2(fname)
