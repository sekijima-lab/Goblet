#!/bin/bash

#for var in akt1; do
for var in akt1 ampc cxcr4 cp3a4 hivrt hivpr gcr kif11; do
    qsub -g $GROUP -M $MAIL -N $var 0_generate.job.sh
done