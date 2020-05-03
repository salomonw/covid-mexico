# covid-mexico  (predictive models) 
Personalized models that predict the following events: (1) hospitalization, (2) mortality, (3) need for ICU, and (4) need for a ventilator, for covid symptomatic patients.

## Publication:
- S. Wolllenstein-Betech, C.G. Cassandras, I.Ch. Paschalidis. 
***"Personalized Predictive Models for Symptomatic COVID-19 Patients Using Basic Preconditions: Hospitalizations, Mortality, and the Need for an ICU or Ventilator."*** (2020).  To appear  


## Install and use:
- For docker users 
    - To build: `sudo docker build . -t covid_mex`
    - To use: `sudo sh run_local.sh` 
- To build local:
    - Requirements:
        - numpy
        - pandas
        - matplotlib
        - seaborn
        - sklearn
        - tensorflow
        - xgboost
        - datetime
    - To use:
        - run `python3 experiments/run_experiment`
- see results on the `results` directory
- to update data file, change lin `267` in `python3 experiments/run_experiment`
## Abstract
*Background:* 
The rapid global spread of the virus SARS-CoV-2 has provoked a spike in demand for hospital care. Hospital systems across the world have been over-extended, including in Northern Italy, Ecuador, and New York City, and many other systems face similar challenges. As a result, decisions on how to best allocate very limited medical resources have come to the forefront. Specifically, under consideration are decisions on who to test, who to admit into hospitals, who to treat in an Intensive Care Unit (ICU), and who to support with a ventilator. Given today’s ability to gather, share, analyze and process data, personalized predictive models based on demographics and information regarding prior conditions can be used to (1) help decision-makers allocate limited resources, when needed, (2) advise individuals how to better protect themselves given their risk profile, (3) differentiate social distancing guidelines based on risk, and (4) prioritize vaccinations once a vaccine becomes available. 

*Objective:* 
To develop personalized models that predict the following events: (1) hospitalization, (2) mortality, (3) need for ICU, and (4) need for a ventilator. To predict hospitalization, it is assumed that one has access to a patient’s basic preconditions, which can be easily gathered without the need to be at a hospital. For the remaining models, different versions developed include different sets of a patient’s features, with some including information on how the disease is progressing (e.g., diagnosis of pneumonia). 

*Materials and Methods:* 
Data from a publicly available repository, updated daily, containing information from approximately 91,000 patients in Mexico were used. The data for each patient include demographics, prior medical conditions, SARS-CoV-2 test results, hospitalization, mortality and whether a patient has developed pneumonia or not. Several classification methods were applied, including robust versions of logistic regression, and support vector machines, as well as random forests and gradient boosted decision trees. 

*Results:* 
Interpretable methods (logistic regression and support vector machines) perform just as well as more complex models in terms of accuracy and detection rates, with the additional benefit of elucidating variables on which the predictions are based. Classification accuracies reached 61%, 76%, 83%, and 84% for predicting hospitalization, mortality, need for ICU and need for a ventilator, respectively. The analysis reveals the most important preconditions for making the predictions. For the four models derived, these are: (1) for hospitalization: age, gender, chronic renal insufficiency, diabetes, immunosuppression; (2) for mortality: age, SARS-CoV-2 test status, immunosuppression and pregnancy; (3) for ICU need: development of pneumonia (if available), cardiovascular disease, asthma, and SARS-CoV-2 test status; and (4) for ventilator need: ICU and pneumonia (if available), age, gender, cardiovascular disease, obesity, pregnancy, and SARS-CoV-2 test result
