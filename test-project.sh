#!/bin/bash

# the output directory for the tests:
OUTPUT_DIR="testoutput"

# activate Python environment
source .venv/bin/activate

## declare an array variable
declare -a SYMBOLS=("AAPL" "DIA" "DOGG" "GOOGL" "IBM" "TSLA" "WMT")

## loop through the above array
for SYMBOL in "${SYMBOLS[@]}"
do
    echo " "
    echo "Testing with symbol ${SYMBOL} ..."
    echo " "
    
    echo "   first run"
    mkdir -p ${OUTPUT_DIR}/${SYMBOL}-run1
    python project.py --symbol "${SYMBOL}" --mode test --file "data/alphavantage_TIME_SERIES_DAILY_ADJUSTED__${SYMBOL}__data.json" &> project-run.log
    mv *.json ${OUTPUT_DIR}/${SYMBOL}-run1
    mv figure*.png ${OUTPUT_DIR}/${SYMBOL}-run1
    mv project-run.log ${OUTPUT_DIR}/${SYMBOL}-run1

    echo "   second run"
    mkdir -p ${OUTPUT_DIR}/${SYMBOL}-run2
    python project.py --symbol "${SYMBOL}" --mode test  --file "data/alphavantage_TIME_SERIES_DAILY_ADJUSTED__${SYMBOL}__data.json" &> project-run.log
    mv *.json ${OUTPUT_DIR}/${SYMBOL}-run2
    mv figure*.png ${OUTPUT_DIR}/${SYMBOL}-run2
    mv project-run.log ${OUTPUT_DIR}/${SYMBOL}-run2

    # there shouldn't be any differences if we run the script in test mode
    echo "   diff'ing run1 and run2 ..."
    diff -r -q ${OUTPUT_DIR}/${SYMBOL}-run1 ${OUTPUT_DIR}/${SYMBOL}-run2
    # rename run1 directory so we can do a folder diff with VSCode's extension "Diff Folders" or similar tools
    mv ${OUTPUT_DIR}/${SYMBOL}-run1 ${OUTPUT_DIR}/${SYMBOL}

    # the new restults should be the same as the ones saved before
    echo "   diff'ing sanctioned-output and run1 ..."
    diff -r -q --exclude=figure*.png --exclude=00_alphavantage_TIME_SERIES_DAILY_ADJUSTED*.json data/sanctioned-output/${SYMBOL} ${OUTPUT_DIR}/${SYMBOL}
done




