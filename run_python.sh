#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_96python3 x86_64-centos7-gcc8-opt

export PYTHONPATH="/afs/cern.ch/user/j/japaszki/blond_sims/blond/:$PYTHONPATH"
export PYTHONPATH="/afs/cern.ch/user/j/japaszki/blond_sims/ps_cb_blond_sims/:$PYTHONPATH"

python /afs/cern.ch/user/j/japaszki/blond_sims/ps_cb_blond_sims/$1