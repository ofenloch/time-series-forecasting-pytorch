#!/bin/bash

# This is a little script generating the original input file that was 
# used to generate the sanctioned output files.

for DIR in ./data/sanctioned-output/*/ # list directories in the form "./data/dirname/"
do
    DIR=${DIR%*/}     # remove the trailing "/"
    DIR="${DIR##*/}"  # cut off everything in front of the final "/"
    SYMBOL=${DIR}     # the directory name is the symbol's name
    echo "processing symbol ${SYMBOL} ..."
    FILENAME="alphavantage_TIME_SERIES_DAILY_ADJUSTED__${SYMBOL}__data.json"
    echo -e "{\n{\n\"Meta Data\":" > ${FILENAME}
    cat ./data/sanctioned-output/${SYMBOL}/00b_meta_data.json >> ${FILENAME}
    echo -e "},\n\"Time Series (Daily)\":\n" >> ${FILENAME}
    cat ./data/sanctioned-output/${SYMBOL}/00a_data.json >> ${FILENAME}
    echo -e "}">> ${FILENAME}
    echo "  generated file ${FILENAME}"
done
