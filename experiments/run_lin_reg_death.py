import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def plot_sns_hists(df, cols):
    n = int(np.sqrt(len(cols)))
    f, axes = plt.subplots(n , (n+1), figsize=(7, 7), sharex=True)
    i=0
    j=0
    for col in cols:
        if i == n:
            i=0
            j+=1
        sns.distplot(df[col], color="skyblue", ax=axes[i,j])
        i+=1
    return f, axes

fname = "../data/COVID19_Mexico_13.04.2020.csv"
percentage_train = 0.7


CATEGORICAL_COLUMNS = [ #'FECHA_ACTUALIZACION',
                        #'ORIGEN',
                        #'SECTOR',
                        #'ENTIDAD_UM',
                        'SEXO',
                        #'ENTIDAD_NAC',
                        #'ENTIDAD_RES',
                        #'MUNICIPIO_RES',
                        #'TIPO_PACIENTE',
                        #'FECHA_INGRESO',
                        #'FECHA_SINTOMAS',
                        #'FECHA_DEF',
                        'INTUBADO',
                        'NEUMONIA',
                        #'EDAD',
                        #'NACIONALIDAD',
                        'EMBARAZO',
                        'HABLA_LENGUA_INDI',
                        'DIABETES',
                        'EPOC',
                        'ASMA',
                        'INMUSUPR',
                        'HIPERTENSION',
                        'OTRA_CON',
                        'CARDIOVASCULAR',
                        'OBESIDAD',
                        'RENAL_CRONICA',
                        'TABAQUISMO',
                        #'OTRO_CASO',
                        #'RESULTADO',
                        #'MIGRANTE',
                        #'PAIS_NACIONALIDAD',
                        #'PAIS_ORIGEN'
                        #'UCI',
                        ]

#CATEGORICAL_COLUMNS = ['SEXO']

NUMERIC_COLUMNS = [     #'FECHA_ACTUALIZACION',
                        #'ORIGEN',
                        #'SECTOR',
                        #'ENTIDAD_UM',
                        #'SEXO',
                        #'ENTIDAD_NAC',
                        #'ENTIDAD_RES',
                        #'MUNICIPIO_RES',
                        #'TIPO_PACIENTE',
                        #'FECHA_INGRESO',
                        #'FECHA_SINTOMAS',
                        #'FECHA_DEF',
                        #'INTUBADO',
                        #'NEUMONIA',
                       'EDAD'
                        #'NACIONALIDAD',
                        #'EMBARAZO',
                        #'HABLA_LENGUA_INDI',
                        #'DIABETES',
                        #'EPOC',
                        #'ASMA',
                        #'INMUSUPR',
                        #'HIPERTENSION',
                        #'OTRA_CON',
                        #'CARDIOVASCULAR',
                        #'OBESIDAD',
                        #'RENAL_CRONICA',
                        #'TABAQUISMO',
                        #'OTRO_CASO',
                        #'RESULTADO',
                        #'MIGRANTE',
                        #'PAIS_NACIONALIDAD',
                        #'PAIS_ORIGEN',
                        #'UCI',
                        ]


#NUMERIC_COLUMNS = []
cols = CATEGORICAL_COLUMNS.copy()
cols.extend(NUMERIC_COLUMNS)
cols.append('TIPO_PACIENTE')

cols_plot= ['EMBARAZO',
            'HABLA_LENGUA_INDI',
            'DIABETES',
            'EPOC',
            'ASMA',
            'INMUSUPR',
            'HIPERTENSION',
            'OTRA_CON',
            'CARDIOVASCULAR',
            'OBESIDAD',
            'RENAL_CRONICA',
            'TABAQUISMO']


df = pd.read_csv(fname, encoding = "ISO-8859-1")
df = df[cols]
#dftrain = df.sample(frac=percentage_train)
#dfeval = pd.concat([df, dftrain]).drop_duplicates(keep=False)

#y_train = dftrain.pop('RESULTADO')
#y_eval = dfeval.pop('RESULTADO')

train_dataset = df.sample(frac=percentage_train,random_state=0)
test_dataset = df.drop(train_dataset.index)

fig, ax = plot_sns_hists(df, cols_plot)
plt.show()

sns.distplot(df['TIPO_PACIENTE'], color="skyblue")
plt.show()

normed_train_data= train_dataset
normed_test_data = test_dataset

def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

model = build_model()

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

