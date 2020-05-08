#!/bin/bash

wget "http://187.191.75.115/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip"
unzip datos_abiertos_covid19.zip -d data
rm datos_abiertos_covid19.zip

python3 experiments/run_experiments.py

git add . -A
git commit -m 'update results'
git push origin master