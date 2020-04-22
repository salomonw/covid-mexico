import data_analysis as DA
import os
from datetime import datetime

# Get timestamp
ts_obj = datetime.now()
ts = ts_obj.strftime("%d-%b-%Y_%H-%M-%S")
os.mkdir('results/'+ts)

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

## Run all data analysis figures
print('Data Analytics started ...')
os.mkdir('results/'+ts+'/DataAnalysis/')
DA.run(fname=fname, out_dir='results/'+ts+'/DataAnalysis')
print('Data Analytics ready!')
