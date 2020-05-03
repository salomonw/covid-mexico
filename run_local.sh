#!/bin/bash
docker run -v `pwd`/data:/data  -v `pwd`/experiments:/experiments  -v `pwd`/results:/results  -t -i covid_mex /bin/bash -c "python3 experiments/run_experiments.py"