#!/bin/bash

# the output directory for the tests:
OUTPUT_DIR="testoutput"

# activate Python environment
source .venv/bin/activate

## declare an array variable
declare -a SYMBOLS=("DIA" "DOGG" "GOOGL" "IBM" "TSLA" "WMT")

## loop through the above array
for SYMBOL in "${SYMBOLS[@]}"
do
    # first run
    mkdir -p ${OUTPUT_DIR}/${SYMBOL}-run1
    python project.py --symbol ${SYMBOL} --mode test &> project-run.log
    mv *.json ${OUTPUT_DIR}/${SYMBOL}-run1
    mv figure*.png ${OUTPUT_DIR}/${SYMBOL}-run1
    mv project-run.log ${OUTPUT_DIR}/${SYMBOL}-run1

    # secont run
    mkdir -p ${OUTPUT_DIR}/${SYMBOL}-run2
    python project.py --symbol ${SYMBOL} --mode test &> project-run.log
    mv *.json ${OUTPUT_DIR}/${SYMBOL}-run2
    mv figure*.png ${OUTPUT_DIR}/${SYMBOL}-run2
    mv project-run.log ${OUTPUT_DIR}/${SYMBOL}-run2

    # there shouldn't be any differences if we run the script in test mode
    diff -r -q ${OUTPUT_DIR}/${SYMBOL}-run1 ${OUTPUT_DIR}/${SYMBOL}-run2

    # the new restults should be the same as the ones saved before
    diff -r -q --exclude=figure*.png data/sanctioned-output/${SYMBOL} ${OUTPUT_DIR}/${SYMBOL}-run1
done




