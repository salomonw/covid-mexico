#!/bin/bash

python3 experiments/run_experiments.py
git add . -A
git commit -m 'update results'
git push origin master