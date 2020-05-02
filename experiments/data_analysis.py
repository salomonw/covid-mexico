import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['determination'] = ssreg / sstot

    return results



def hisogram_per_age(df, age_low, age_high, color):
    df21 = df[(df.EDAD >= age_low) & (df.EDAD < age_high)]
    x = df21.F.dt.days
    ax = plt.hist(df21.F.dt.days, label=str(age_low) + '-' + str(age_high) + ' years', bins=range(0, 30), alpha=0.9,
             color=color)
    return ax

def run(fname, out_dir):

    df = pd.read_csv(fname, encoding = "ISO-8859-1")
    df.FECHA_INGRESO = pd.to_datetime(df.FECHA_INGRESO)
    df.FECHA_SINTOMAS = pd.to_datetime(df.FECHA_SINTOMAS)
    df = df.replace("9999-99-99","2022-04-05")
    df.FECHA_DEF = pd.to_datetime(df.FECHA_DEF)

    df0 = df.copy()
    df0 = df0[df0.EDAD < 101]

    df = df[(df.RESULTADO==1) | (df.RESULTADO==3)]
    df = df[df.EDAD<101]

    outlier = df[(df.EMBARAZO == 1) & (df.EDAD > 55)]

    df = df.merge(outlier, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']



    ## Create Data Summary
    summary = {}

    summary['Tests'] = {}
    summary['Tests']['m'] = len(df0)
    summary['Tests']['Y'] = len(df0[(df0.RESULTADO==1)])
    summary['Tests']['W'] = len(df0[(df0.RESULTADO == 3)])
    summary['Tests']['N'] = len(df0[(df0.RESULTADO == 2)])

    summary['Hospitalization'] = {}
    summary['Hospitalization']['m'] = len(df0[(df0.TIPO_PACIENTE==2)])
    summary['Hospitalization']['Y'] = len(df0[(df0.TIPO_PACIENTE == 2) & (df0.RESULTADO == 1) ])
    summary['Hospitalization']['W'] = len(df0[(df0.TIPO_PACIENTE == 2) & (df0.RESULTADO == 3)])
    summary['Hospitalization']['N'] = len(df0[(df0.TIPO_PACIENTE == 2) & (df0.RESULTADO == 2)])
    summary['Hospitalization']['Pneu'] = len(df0[(df0.TIPO_PACIENTE == 2) & (df0.NEUMONIA == 1)])
    summary['Hospitalization']['Vent'] = len(df0[(df0.TIPO_PACIENTE == 2) & (df0.INTUBADO == 1)])
    summary['Hospitalization']['ICU'] = len(df0[(df0.TIPO_PACIENTE == 2) & (df0.UCI == 1)])

    summary['Characteristics'] = {}
    summary['Characteristics']['Pregnant']= len(df0[(df0.EMBARAZO == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['Diabetes']= len(df0[(df0.DIABETES == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['COPD']=     len(df0[(df0.EPOC == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['Asthma']=   len(df0[(df0.ASMA == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['Immunosuppression']=    len(df0[(df0.INMUSUPR == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['Hypertension']=         len(df0[(df0.HIPERTENSION == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['Other']=                len(df0[(df0.OTRA_COM == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['Cardiovascular disease']=len(df0[(df0.CARDIOVASCULAR == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['Obesity']=               len(df0[(df0.OBESIDAD == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['Chronic renal insufficiency']=len(df0[(df0.RENAL_CRONICA == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['Tobacco Use']=              len(df0[(df0.TABAQUISMO == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['Contact COVID case'] = len(df0[(df0.OTRO_CASO == 1) & (df0.RESULTADO != 2)])
    summary['Characteristics']['Speak indigenous len'] = len(df0[(df0.HABLA_LENGUA_INDIG == 1) & (df0.RESULTADO != 2)])

    print(summary)
    summary_df = pd.DataFrame.from_dict(summary)
    summary_df = summary_df.T
    summary_df.to_latex(out_dir + '/summary.tex', float_format="%i")



    ## Cuanto tiempo tarda en responder la gente?
    plt.figure(figsize=(4,4))
    df2 = df.copy()
    df2['F'] = df.FECHA_INGRESO - df.FECHA_SINTOMAS
    df2 = df2[(df2.F.dt.days>=0) & (df2.F.dt.days<=30)]
    ax = hisogram_per_age(df2, 30, 50, color='orange')
    ax = hisogram_per_age(df2, 50, 100, color='blue')
    ax = hisogram_per_age(df2, 0, 30, color='g')
    plt.legend()
    plt.title('Admission date - Symptoms date')
    plt.ylabel('Counts')
    plt.xlabel('days')
    plt.tight_layout()
    plt.savefig(out_dir + '/response_to_sym_time.png')
    plt.savefig(out_dir + '/response_to_sym_time.pdf')
    plt.close('all')

    ## Cuanto tarda la gente de sintomas a muerte?
    plt.figure(figsize=(4, 4))
    df2['F'] = df.FECHA_DEF  - df.FECHA_SINTOMAS
    df2 = df2[df2.F.dt.days>=0]
    df2 = df2[df2.F.dt.days<=60]
    print(df2.F.dt.days.mean())
    print(np.sqrt(df2.F.dt.days.var()))
    ax = hisogram_per_age(df2, 50, 100, color='blue')
    ax = hisogram_per_age(df2, 30, 50, color='orange')
    ax = hisogram_per_age(df2, 0, 30, color='g')
    plt.title('Death date - Symptoms date')
    plt.ylabel('Count, '  + str(len(df2)) + ' data points')
    plt.xlabel('days')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir + '/sym_to_die_time.png')
    plt.savefig(out_dir + '/sym_to_die_time.pdf')
    plt.close('all')

    ## Cuanto tarda la gente de admision a muerte?
    plt.figure(figsize=(4, 4))
    df2['F'] = df.FECHA_DEF  - df.FECHA_INGRESO
    df2 = df2[df2.F.dt.days>=0]
    df2 = df2[df2.F.dt.days<=60]
    ax = hisogram_per_age(df2, 50, 100, color='blue')
    ax = hisogram_per_age(df2, 30, 50, color='orange')
    ax = hisogram_per_age(df2, 0, 30, color='g')
    plt.title('Death date - Admission date')
    plt.ylabel('Counts')#, '  + str(len(df2)) + ' data points')
    plt.xlabel('days')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir + '/hospital_time.png')
    plt.savefig(out_dir + '/hospital_time.pdf')
    plt.close('all')


    features_of_interest = [   # 'FECHA_ACTUALIZACION',
                               # 'ORIGEN',
                               # 'SECTOR',
                               # 'ENTIDAD_UM',
                                'SEXO',
                               # 'ENTIDAD_NAC',
                               # 'ENTIDAD_RES',
                               # 'MUNICIPIO_RES',
                                #'TIPO_PACIENTE',
                               # 'FECHA_INGRESO',
                               # 'FECHA_SINTOMAS',
                               # 'FECHA_DEF',
                             #   'INTUBADO',
                                'NEUMONIA',
                               # 'EDAD',
                               # 'NACIONALIDAD',
                                'EMBARAZO',
                              #  'HABLA_LENGUA_INDIG',
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
                              #  'UCI',
                                ]

    feature_translate = {
            'SEXO': 'Gender ',
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
    edad_list = []
    percentage_hos_list = []
    edades = list(np.linspace(0,100,51))
    for j in range(len(edades)-1):# sorted(list(df.EDAD.unique())):
        num = len(df[(df.EDAD >= edades[j]) & (df.EDAD < edades[j+1]) &(df.TIPO_PACIENTE==2)])
        den = len(df[(df.EDAD >= edades[j]) & (df.EDAD < edades[j+1])])
        ax[0].bar(x=edades[j+1] , height=num/den, color='blue')
        plt.grid(True)
        ax[1].scatter(x=edades[j+1], y=den, marker='*', color='red')
        plt.grid(True)
        edad_list.append(edades[j+1])
        percentage_hos_list.append(num / den)
    ax[0].set(ylabel="Percentage being Hospitalized")

    ## plot wih linear regression :
    x = edad_list[-35:][:20]
    y = percentage_hos_list[-35:][:20]
    results = polyfit(x, y, 1)
    [m,b]= results['polynomial']
    r2 = results['determination']
    yp = np.polyval([m, b], x)
    ax[0].plot(x, yp, color='orangered', label=str(round(b,2))+'+' + str(round(m,4))+'*Age; R2='+str(round(r2,2)))
    ax[0].legend()
    ax[1].set(ylabel="Counts")
    plt.xlabel('Age')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir + '/hospitalization_age.png')
    plt.savefig(out_dir + '/hospitalization_age.pdf')
    plt.close('all')




    # Table being hospitalized
    db1 = []
    db2 = []
    db3 = []
    db4 = []
    db5 = []
    features=[]
    for feature in features_of_interest:
        if feature == 'SEXO':
            Ni = len(df[df[feature] == 1])
            db1.append(len(df[(df[feature] == 1) & (df['TIPO_PACIENTE'] == 2)]) / Ni)
            db2.append(len(df[(df[feature] == 1) & ((df['FECHA_DEF'] - df['FECHA_INGRESO']).dt.days <= 100)]) / Ni)
            db3.append(len(df[(df[feature] == 1) & (df['UCI'] == 1)]) / Ni)
            db4.append(len(df[(df[feature] == 1) & (df['INTUBADO'] == 1)]) / Ni)
            #db5.append(len(df[(df[feature] == 1) & (df['NEUMONIA'] == 1)]) / Ni)
            features.append('Female' + ' (' + str(Ni) + ')')

            Ni = len(df[df[feature] == 2])
            db1.append(len(df[(df[feature] == 1) & (df['TIPO_PACIENTE'] == 2)]) / Ni)
            db2.append(len(df[(df[feature] == 1) & ((df['FECHA_DEF'] - df['FECHA_INGRESO']).dt.days <= 100)]) / Ni)
            db3.append(len(df[(df[feature] == 1) & (df['UCI'] == 1)]) / Ni)
            db4.append(len(df[(df[feature] == 1) & (df['INTUBADO'] == 1)]) / Ni)
            #db5.append(len(df[(df[feature] == 1) & (df['NEUMONIA'] == 1)]) / Ni)
            features.append('Male' + ' (' + str(Ni) + ')')
        else:
            Ni = len(df[df[feature] == 1])
            db1.append(len(df[(df[feature] == 1) & (df['TIPO_PACIENTE'] == 2)]) / Ni)
            db2.append(len(df[(df[feature] == 1) & ((df['FECHA_DEF'] - df['FECHA_INGRESO']).dt.days <= 100)]) / Ni)
            db3.append(len(df[(df[feature] == 1) & (df['UCI'] == 1)]) / Ni)
            db4.append(len(df[(df[feature] == 1) & (df['INTUBADO'] == 1)]) / Ni)
            #db5.append(len(df[(df[feature] == 1) & (df['NEUMONIA'] == 1)]) / Ni)
            features.append(feature_translate[feature] + ' (' + str(Ni) + ')')

    df_f = pd.DataFrame({'Hospitalized': db1,  'Death': db2,  'ICU': db3,  'Ventilator': db4}, index=features)
    df_f = df_f.sort_values('Hospitalized', ascending = False)
    df_f_transpose = df_f#.transpose()
    plt.figure(figsize = (8, 6))
    g= sns.heatmap(df_f_transpose,
                annot=True,
                #square=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    #g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    #g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_features.png')
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
    plt.savefig(out_dir + '/hosp_features2.png')
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
    plt.savefig(out_dir + '/hosp_features3.png')
    plt.savefig(out_dir + '/hosp_features3.pdf')
    plt.close('all')


    # Table Edad Test positive
    ages = np.arange(0,105,5)
    dic = {}
    for i in range(len(ages)-1):
        df2 = df[(df.EDAD>ages[i]) & (df.EDAD<=ages[i+1])]
        d0 = df[(df.EDAD>ages[i]) & (df.EDAD<=ages[i+1])]
        db1 = []
        db2 = []
        features_list = []
        for feature in features_of_interest:
            features_list.append(feature_translate[feature])
            Ni = len(df2[df2[feature] == 1])
            if feature != 'SEXO':
                d0 = d0[d0[feature] !=1]
            if Ni < 10:
                db1.append(np.nan)
                db2.append(np.nan)
            else:
                db1.append(len(df2[(df2[feature] == 1) & (df2['TIPO_PACIENTE'] == 2)]) / Ni)
                db2.append(len(df2[(df2[feature]==1) & ((df2['FECHA_DEF']-df2['FECHA_INGRESO']).dt.days <=100)])/Ni)

        N0 = len(d0)
        db1.append(len(d0[d0['TIPO_PACIENTE'] == 2]) / N0)
        dic[str(ages[i]) + '-' + str(ages[i+1])] = db1

    features_list.append('No preconditions')
    df_f = pd.DataFrame(dic, index=features_list)
    g = sns.set(font_scale=0.7)
    plt.figure(figsize = (15*.7,9*.7))
    g = sns.heatmap(df_f,
                annot=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    g.invert_yaxis()
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_age_features1.png')
    plt.savefig(out_dir + '/hosp_age_features1.pdf')
    plt.close('all')



    # Table Edad Test positive
    ages = np.arange(0,100,10)
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
    plt.figure(figsize = (15*.7,9*.7))
    g = sns.heatmap(df_f,
                annot=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    g.invert_yaxis()
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_age_features2.png')
    plt.savefig(out_dir + '/hosp_age_features2.pdf')
    plt.close('all')



    # Table Edad Test positive
    ages = np.arange(0,100,20)
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
    plt.figure(figsize = (15*.7,9*.7))
    g = sns.heatmap(df_f,
                annot=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    g.invert_yaxis()
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_age_features3.pdf')
    plt.savefig(out_dir + '/hosp_age_features3.png')
    plt.close('all')