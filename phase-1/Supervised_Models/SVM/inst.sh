#!/bin/bash
#BSUB -q q_hpc 
#BSUB -n 1
#BSUB -m g1
#BSUB -M 32768
#BSUB -oo salida
#BSUB -eo error
python instructions.py 