#!/bin/bash

# the output directory for the tests:
OUTPUT_DIR="testoutput"

# activate Python environment
source .venv/bin/activate


# first run
mkdir -p ${OUTPUT_DIR}/run1
python project.py &> project-run.log
mv *.json ${OUTPUT_DIR}/run1
mv project-run.log ${OUTPUT_DIR}/run1

# secont run
mkdir -p ${OUTPUT_DIR}/run2
python project.py &> project-run.log
mv *.json ${OUTPUT_DIR}/run2
mv project-run.log ${OUTPUT_DIR}/run2

# there shouldn't be any differences if we run the script in test mode
diff -r -q ${OUTPUT_DIR}/run1 ${OUTPUT_DIR}/run2