import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def run(fname, out_dir):

    df = pd.read_csv(fname, encoding = "ISO-8859-1")
    df.FECHA_INGRESO = pd.to_datetime(df.FECHA_INGRESO)
    df.FECHA_SINTOMAS = pd.to_datetime(df.FECHA_SINTOMAS)
    df = df.replace("9999-99-99","2022-04-05")
    df.FECHA_DEF = pd.to_datetime(df.FECHA_DEF)

    df0 = df.copy()
    df = df[(df.RESULTADO==1) | (df.RESULTADO==3)]

    ## Create Data Summary
    summary = {}

    summary['Tests'] = {}
    summary['Tests']['m'] = len(df0)
    summary['Tests']['Y+W'] = len(df)
    summary['Tests']['Y'] = len(df0[(df0.RESULTADO==1)])

    summary['Hospitalization'] = {}
    summary['Hospitalization']['m'] = len(df0[(df0.TIPO_PACIENTE==2)])
    summary['Hospitalization']['Y'] = len(df0[(df0.TIPO_PACIENTE == 2) & (df0.RESULTADO == 1) ])
    summary['Hospitalization']['W'] = len(df0[(df0.TIPO_PACIENTE == 2) & (df0.RESULTADO == 3)])
    summary['Hospitalization']['Pneu'] = len(df0[(df0.TIPO_PACIENTE == 2) & (df0.NEUMONIA == 1)])
    summary['Hospitalization']['Vent'] = len(df0[(df0.TIPO_PACIENTE == 2) & (df0.INTUBADO == 1)])
    summary['Hospitalization']['ICU'] = len(df0[(df0.TIPO_PACIENTE == 2) & (df0.UCI == 1)])

    summary['Characteristics'] = {}
    summary['Characteristics']['Pregnant']= len(df0[(df0.EMBARAZO == 1)])
    summary['Characteristics']['Diabetes']= len(df0[(df0.DIABETES == 1)])
    summary['Characteristics']['COPD']=     len(df0[(df0.EPOC == 1)])
    summary['Characteristics']['Asthma']=   len(df0[(df0.ASMA == 1)])
    summary['Characteristics']['Immunosuppression']=    len(df0[(df0.INMUSUPR == 1)])
    summary['Characteristics']['Hypertension']=         len(df0[(df0.HIPERTENSION == 1)])
    summary['Characteristics']['Other']=                len(df0[(df0.OTRA_COM == 1)])
    summary['Characteristics']['Cardiovascular disease']=len(df0[(df0.CARDIOVASCULAR == 1)])
    summary['Characteristics']['Obesity']=               len(df0[(df0.OBESIDAD == 1)])
    summary['Characteristics']['Chronic renal insufficiency']=len(df0[(df0.RENAL_CRONICA == 1)])
    summary['Characteristics']['Tobacco Use']=              len(df0[(df0.TABAQUISMO == 1)])
    summary['Characteristics']['Contact COVID case'] = len(df0[(df0.OTRO_CASO == 1)])
    summary['Characteristics']['Speak indigenous len'] = len(df0[(df0.HABLA_LENGUA_INDIG == 1)])

    summary_df = pd.DataFrame.from_dict(summary)
    summary_df = summary_df.T
    summary_df.to_latex(out_dir + '/summary.tex', float_format="%i")



    ## Cuanto tiempo tarda en responder la gente?
    df2 = df.FECHA_INGRESO  - df.FECHA_SINTOMAS
    df2 = df2[df2.dt.days>=0]
    df2 = df2[df2.dt.days<=30]
    df2.dt.days.hist(bins=30)
    plt.title('Response Time (Admission date - Symptoms date)')
    plt.ylabel('Count, '  + str(len(df2)) + ' data points')
    plt.xlabel('days')
    plt.tight_layout()
    plt.savefig(out_dir + '/response_to_sym_time.pdf')
    plt.close('all')

    ## Cuanto tarda la gente de sintomas a muerte?
    df2 = df.FECHA_DEF  - df.FECHA_SINTOMAS
    df2 = df2[df2.dt.days>=0]
    df2 = df2[df2.dt.days<=60]
    df2.dt.days.hist(bins=30)
    plt.title('Death date - Symptoms date')
    plt.ylabel('Count, '  + str(len(df2)) + ' data points')
    plt.xlabel('days')
    plt.tight_layout()
    plt.savefig(out_dir + '/sym_to_die_time.pdf')
    plt.close('all')

    ## Cuanto tarda la gente de admision a muerte?
    df2 = df.FECHA_DEF  - df.FECHA_INGRESO
    df2 = df2[df2.dt.days>=0]
    df2 = df2[df2.dt.days<=60]
    df2.dt.days.hist(bins=30)
    plt.title('Death date - Admission date')
    plt.ylabel('Count, '  + str(len(df2)) + ' data points')
    plt.xlabel('days')
    plt.tight_layout()
    plt.savefig(out_dir + '/hospital_time.pdf')
    plt.close('all')


    features_of_interest = [   # 'FECHA_ACTUALIZACION',
                               # 'ORIGEN',
                               # 'SECTOR',
                               # 'ENTIDAD_UM',
                               # 'SEXO',
                               # 'ENTIDAD_NAC',
                               # 'ENTIDAD_RES',
                               # 'MUNICIPIO_RES',
                                #'TIPO_PACIENTE',
                               # 'FECHA_INGRESO',
                               # 'FECHA_SINTOMAS',
                               # 'FECHA_DEF',
                                'INTUBADO',
                                'NEUMONIA',
                               # 'EDAD',
                               # 'NACIONALIDAD',
                                'EMBARAZO',
                                'HABLA_LENGUA_INDIG',
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
                                'OTRO_CASO',
                               # 'RESULTADO',
                               # 'MIGRANTE',
                               # 'PAIS_NACIONALIDAD',
                               # 'PAIS_ORIGEN'
                                'UCI',
                                ]

    feature_translate = {
            'EMBARAZO': 'Pregnant ',
            'DIABETES': 'Diabetes ',
            'EPOC': 'COPD',
            'ASMA': 'Asthma ',
            'INMUSUPR': 'Immunosuppression ',
            'HIPERTENSION': 'Hypertension ',
            'OTRA_COM': 'Other  ',
            'OTRA_CON': 'Other  ',
            'CARDIOVASCULAR': 'Cardiovascular disease ',
            'OBESIDAD': 'Obesity ',
            'RENAL_CRONICA': 'Chronic renal insufficiency ',
            'TABAQUISMO': 'Tobacco use ',
            'OTRO_CASO': 'Contact COVID case ',
            'NEUMONIA': 'Pneumonia ',
            'INTUBADO': 'Need a ventilator ',
            'UCI': 'ICU ',
            'HABLA_LENGUA_INDIG' : 'Speak indigenous len'
    }


    ## Hospitalization per age and number of samples
    N = len(df)
    fig, ax = plt.subplots(2)
    for edad in df.EDAD.unique():
        num = len(df[(df.EDAD == edad) & (df.TIPO_PACIENTE==2)])
        den = len(df[df.EDAD == edad])
        ax[0].bar(x=edad , height=num/den, color='blue')
        plt.grid(True)
        ax[1].scatter(x=edad, y=den, marker='*', color='red')
        plt.grid(True)
    ax[0].set(ylabel="Percentage being Hospitalized")
    ax[1].set(ylabel="Counts")
    plt.xlabel('Age')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir + '/hospitalization_age.pdf')
    plt.close('all')


    # Table being hospitalized
    db1 = []
    db2 = []
    features=[]
    for feature in features_of_interest:
        Ni = len(df[df[feature]==1])
        db1.append(len(df[(df[feature]==1) & (df['TIPO_PACIENTE']==2)])/Ni)
        db2.append(len(df[(df[feature]==1) & ((df['FECHA_DEF']-df['FECHA_INGRESO']).dt.days <=100)])/Ni)
        features.append(feature_translate[feature]+' ('+str(Ni)+')')
    df_f = pd.DataFrame({'Hospitalized': db1,  'Death': db2}, index=features)
    df_f = df_f.sort_values('Hospitalized', ascending = False)
    df_f_transpose = df_f.transpose()
    plt.figure(figsize = (14, 3))
    g= sns.heatmap(df_f_transpose,
                annot=True,
                #square=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_features.pdf')
    plt.close('all')


    del df_f
    df_f = pd.DataFrame({'Hospitalized': db1}, index=features)
    df_f = df_f.sort_values('Hospitalized', ascending = False)
    df_f_transpose = df_f.transpose()
    plt.figure(figsize = (14, 3))
    g= sns.heatmap(df_f_transpose,
                annot=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    g.invert_yaxis()
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_features2.pdf')
    plt.close('all')


    del df_f
    df_f = pd.DataFrame({'Death': db2}, index=features)
    df_f = df_f.sort_values('Death', ascending = False)
    df_f_transpose = df_f.transpose()
    plt.figure(figsize = (14, 3))
    g= sns.heatmap(df_f_transpose,
                annot=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    g.invert_yaxis()
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_features3.pdf')
    plt.close('all')


    # Table Edad Test positive
    ages = np.arange(0,120,5)
    dic = {}
    for i in range(len(ages)-1):
        df2 = df[(df.EDAD>ages[i]) & (df.EDAD<=ages[i+1])]
        db1 = []
        db2 = []
        features_list = []
        for feature in features_of_interest:
            features_list.append(feature_translate[feature])
            Ni = len(df2[df2[feature] == 1])
            if Ni == 0:
                db1.append(np.nan)
                db2.append(np.nan)
            else:
                db1.append(len(df2[(df2[feature] == 1) & (df2['TIPO_PACIENTE'] == 2)]) / Ni)
                db2.append(len(df2[(df2[feature]==1) & ((df2['FECHA_DEF']-df2['FECHA_INGRESO']).dt.days <=100)])/Ni)
        dic[str(ages[i]) + '-' + str(ages[i+1])] = db1
    df_f = pd.DataFrame(dic, index=features_list)
    g = sns.set(font_scale=0.7)
    plt.figure(figsize = (15,9))
    g = sns.heatmap(df_f,
                annot=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    g.invert_yaxis()
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_age_features1.pdf')
    plt.close('all')



    # Table Edad Test positive
    ages = np.arange(0,120,10)
    dic = {}
    for i in range(len(ages)-1):
        df2 = df[(df.EDAD>ages[i]) & (df.EDAD<=ages[i+1])]
        db1 = []
        db2 = []
        features_list = []
        for feature in features_of_interest:
            features_list.append(feature_translate[feature])
            Ni = len(df2[df2[feature] == 1])
            if Ni == 0:
                db1.append(np.nan)
                db2.append(np.nan)
            else:
                db1.append(len(df2[(df2[feature] == 1) & (df2['TIPO_PACIENTE'] == 2)]) / Ni)
                db2.append(len(df2[(df2[feature]==1) & ((df2['FECHA_DEF']-df2['FECHA_INGRESO']).dt.days <=100)])/Ni)
        dic[str(ages[i]) + '-' + str(ages[i+1])] = db1
    df_f = pd.DataFrame(dic, index=features_list)
    sns.set(font_scale=0.7)
    plt.figure(figsize = (15,9))
    g = sns.heatmap(df_f,
                annot=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    g.invert_yaxis()
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_age_features2.pdf')
    plt.close('all')



    # Table Edad Test positive
    ages = np.arange(0,120,20)
    dic = {}
    for i in range(len(ages)-1):
        df2 = df[(df.EDAD>ages[i]) & (df.EDAD<=ages[i+1])]
        db1 = []
        db2 = []
        features_list = []
        for feature in features_of_interest:
            features_list.append(feature_translate[feature])
            Ni = len(df2[df2[feature] == 1])
            if Ni == 0:
                db1.append(np.nan)
                db2.append(np.nan)
            else:
                db1.append(len(df2[(df2[feature] == 1) & (df2['TIPO_PACIENTE'] == 2)]) / Ni)
                db2.append(len(df2[(df2[feature]==1) & ((df2['FECHA_DEF']-df2['FECHA_INGRESO']).dt.days <=100)])/Ni)
        dic[str(ages[i]) + '-' + str(ages[i+1])] = db1
    df_f = pd.DataFrame(dic, index=features_list)
    sns.set(font_scale=0.7)
    plt.figure(figsize = (15,9))
    g = sns.heatmap(df_f,
                annot=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    g.invert_yaxis()
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_age_features3.pdf')
    plt.close('all')