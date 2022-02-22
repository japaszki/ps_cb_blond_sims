#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_96python3 x86_64-centos7-gcc8-opt

export PYTHONPATH="/eos/user/j/japaszki/blond_sims/BLonD/:$PYTHONPATH"
export PYTHONPATH="/eos/user/j/japaszki/blond_sims/ps_cb_blond_sims/:$PYTHONPATH"

python /eos/user/j/japaszki/blond_sims/ps_cb_blond_sims/run.py

tar -zcvf results.tar.gz plots.tar.gz sim_outputs