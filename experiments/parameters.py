import pandas as pd


def read_data(fname, y, percentage_train=0.7, additional_vars=False, waiting=True):
    CATEGORICAL_COLUMNS = [#'FECHA_ACTUALIZACION',
                           # 'ORIGEN',
                           # 'SECTOR',
                           # 'ENTIDAD_UM',
                            'SEXO',
                           # 'ENTIDAD_NAC',
                           # 'ENTIDAD_RES',
                           # 'MUNICIPIO_RES',
                           # 'TIPO_PACIENTE',
                            #'FECHA_INGRESO',
                            #'FECHA_SINTOMAS',
                            #'FECHA_DEF',
                           # 'INTUBADO',
                           # 'NEUMONIA',
                            #'EDAD',
                           # 'NACIONALIDAD',
                            'EMBARAZO',
                            #'HABLA_LENGUA_INDIG',
                            'DIABETES',
                            'EPOC',
                            'ASMA',
                            'INMUSUPR',
                            'HIPERTENSION',
                            'OTRA_COM',
                            'CARDIOVASCULAR',
                            'OBESIDAD',
                            'RENAL_CRONICA',
                            'TABAQUISMO',
                           # 'OTRO_CASO',
                            #'RESULTADO',
                           # 'MIGRANTE',
                            #'PAIS_NACIONALIDAD',
                           # 'PAIS_ORIGEN'
                            #'UCI',
                            ]
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

    df = pd.read_csv(fname, encoding="ISO-8859-1")

    df = df[df['TIPO_PACIENTE'] < 3]
    df['TIPO_PACIENTE'] = df['TIPO_PACIENTE'] - 1

    if y=='UCI':
        CATEGORICAL_COLUMNS.append('NEUMONIA')
        CATEGORICAL_COLUMNS.append('RESULTADO')
        CATEGORICAL_COLUMNS.append('TIPO_PACIENTE')
        df['UCI'] = df['UCI'].replace(97, 0)
        df['UCI'] = df['UCI'].replace(98, 0)
        df['UCI'] = df['UCI'].replace(99, 0)
        df = df[df['UCI'] < 3]
        df['UCI'] = df['UCI'].replace(2,0)
    elif y=='INTUBADO':
        CATEGORICAL_COLUMNS.append('NEUMONIA')
        CATEGORICAL_COLUMNS.append('RESULTADO')
        CATEGORICAL_COLUMNS.append('TIPO_PACIENTE')
        df['INTUBADO'] = df['INTUBADO'].replace(97, 0)
        df['INTUBADO'] = df['INTUBADO'].replace(98, 0)
        df['INTUBADO'] = df['INTUBADO'].replace(99, 0)
        df = df[df['INTUBADO'] < 3]
        df['INTUBADO'] = df['INTUBADO'].replace(2,0)

    df.FECHA_INGRESO = pd.to_datetime(df.FECHA_INGRESO)
    df.FECHA_SINTOMAS = pd.to_datetime(df.FECHA_SINTOMAS)
    df = df.replace("9999-99-99", "2022-04-05")
    df.FECHA_DEF = pd.to_datetime(df.FECHA_DEF)
    df['MUERTE'] = 0
    df.loc[(df['FECHA_DEF']-df['FECHA_INGRESO']).dt.days <=100, 'MUERTE'] = 1

    for i in CATEGORICAL_COLUMNS:
        df[i] = df[i].replace(2,0)

    if additional_vars != False:
        for vars in additional_vars:
            CATEGORICAL_COLUMNS.append(vars)

    cols = CATEGORICAL_COLUMNS.copy()
    cols.extend(NUMERIC_COLUMNS)
    cols.append(y)

    if waiting == True:
        df = df[(df['RESULTADO'] == 1) | (df['RESULTADO'] == 3)]
    else:
        df = df[(df['RESULTADO'] == 1)]

    df = df[cols]
    df.EDAD = df.EDAD/df.EDAD.max()
    X_train = df.sample(frac=percentage_train)
    X_test = pd.concat([df, X_train]).drop_duplicates(keep=False)
    y_train = X_train.pop(y)
    y_test = X_test.pop(y)

    return [X_train, X_test, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS]


def read_data_eng(fname, y, percentage_train=0.7, additional_vars=False, waiting=True):
    CATEGORICAL_COLUMNS = [#'FECHA_ACTUALIZACION',
                           # 'ORIGEN',
                           # 'SECTOR',
                           # 'ENTIDAD_UM',
                            'SEXO',
                           # 'ENTIDAD_NAC',
                           # 'ENTIDAD_RES',
                           # 'MUNICIPIO_RES',
                           # 'TIPO_PACIENTE',
                            #'FECHA_INGRESO',
                            #'FECHA_SINTOMAS',
                            #'FECHA_DEF',
                           # 'INTUBADO',
                           # 'NEUMONIA',
                            #'EDAD',
                           # 'NACIONALIDAD',
                            'EMBARAZO',
                            #'HABLA_LENGUA_INDI',
                            'DIABETES',
                            'EPOC',
                            'ASMA',
                            'INMUSUPR',
                            'HIPERTENSION',
                            'OTRA_COM',
                            'CARDIOVASCULAR',
                            'OBESIDAD',
                            'RENAL_CRONICA',
                            'TABAQUISMO',
                           # 'OTRO_CASO',
                            #'RESULTADO',
                           # 'MIGRANTE',
                            #'PAIS_NACIONALIDAD',
                           # 'PAIS_ORIGEN'
                            #'UCI',
                            ]
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

    df = pd.read_csv(fname, encoding="ISO-8859-1")

    df = df[df['TIPO_PACIENTE'] < 3]
    df['TIPO_PACIENTE'] = df['TIPO_PACIENTE'] - 1

    if y=='UCI':
        CATEGORICAL_COLUMNS.append('NEUMONIA')
        CATEGORICAL_COLUMNS.append('RESULTADO')
        CATEGORICAL_COLUMNS.append('TIPO_PACIENTE')
        df['UCI'] = df['UCI'].replace(97, 0)
        df['UCI'] = df['UCI'].replace(98, 0)
        df['UCI'] = df['UCI'].replace(99, 0)
        df = df[df['UCI'] < 3]
        df['UCI'] = df['UCI'].replace(2,0)

    elif y=='INTUBADO':
        CATEGORICAL_COLUMNS.append('NEUMONIA')
        CATEGORICAL_COLUMNS.append('RESULTADO')
        CATEGORICAL_COLUMNS.append('TIPO_PACIENTE')
        df['INTUBADO'] = df['INTUBADO'].replace(97, 0)
        df['INTUBADO'] = df['INTUBADO'].replace(98, 0)
        df['INTUBADO'] = df['INTUBADO'].replace(99, 0)
        df = df[df['INTUBADO'] < 3]
        df['INTUBADO'] = df['INTUBADO'].replace(2,0)

    df.FECHA_INGRESO = pd.to_datetime(df.FECHA_INGRESO)
    df.FECHA_SINTOMAS = pd.to_datetime(df.FECHA_SINTOMAS)
    df = df.replace("9999-99-99", "2022-04-05")
    df.FECHA_DEF = pd.to_datetime(df.FECHA_DEF)
    df['MUERTE'] = 0
    df.loc[(df['FECHA_DEF']-df['FECHA_INGRESO']).dt.days <=100, 'MUERTE'] = 1

    for i in CATEGORICAL_COLUMNS:
        df[i] = df[i].replace(2,0)

    if additional_vars != False:
        for vars in additional_vars:
            CATEGORICAL_COLUMNS.append(vars)

    cols = CATEGORICAL_COLUMNS.copy()
    cols.extend(NUMERIC_COLUMNS)
    cols.append(y)

    if waiting == True:
        df = df[(df['RESULTADO'] == 1) | (df['RESULTADO'] == 3)]
    else:
        df = df[(df['RESULTADO'] == 1)]

    df = df[cols]
    df.EDAD = df.EDAD/df.EDAD.max()

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
        'HABLA_LENGUA_INDI': 'Indigenous Len',
        'TIPO_PACIENTE': 'Hospitalization',
        'MUERTE': 'Death',
        'RESULTADO': 'Test',
        'EDAD': 'Age',
        'COPD': 'Age'
    }

    df.rename(columns=feature_translate, inplace=True)
    X_train = df.sample(frac=percentage_train)
    X_test = pd.concat([df, X_train]).drop_duplicates(keep=False)
    y_train = X_train.pop(feature_translate[y])
    y_test = X_test.pop(feature_translate[y])

    return [X_train, X_test, y_train, y_test, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS]
