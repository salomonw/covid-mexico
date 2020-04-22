
import parameters as parameters
import models as models
import pandas as pd
import os
from datetime import datetime


def run_feature_elimination(y, fname, name, out_dir, waiting , add_vars=False):
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

    data = parameters.read_data(fname=fname, y=y,
                                percentage_train=0.7,
                                additional_vars=False,
                                waiting=True)

    [dftrain, dftest, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS] = data

    models.run_recursive_feature_elimination(dftrain=dftrain,
                                             ytrain=y_train,
                                             y=y,
                                             CATEGORICAL_COLUMNS=CATEGORICAL_COLUMNS,
                                             NUMERIC_COLUMNS=NUMERIC_COLUMNS,
                                             name=name,
                                             out_dir=out_dir)






# Get timestamp
ts_obj = datetime.now()
ts = ts_obj.strftime("%d-%b-%Y_%H-%M-%S")
ts+='_FE'
os.mkdir('results/'+ts)

# Specify file name
fname = "data/200419COVID19MEXICO.csv"


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


print('Feature Elimination started ...')
os.mkdir('results/'+ts+'/FeatureElimination/')

for y in ['TIPO_PACIENTE', 'MUERTE', 'UCI', 'INTUBADO']:
    out_dir = 'results/' + ts + '/FeatureElimination/' + y
    os.mkdir(out_dir)
    for waiting in [True, False]:
        out_dir += '/'
        if waiting == True:
            name = 'FE_plot_Y-W'
        else:
            name = 'FE_plot_Y'

        run_feature_elimination(y, fname, name=name, out_dir=out_dir, waiting=waiting , add_vars=False)


print('Feature Elimination ready!')


