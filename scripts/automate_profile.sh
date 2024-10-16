#!/bin/bash

# Define directories
data_dir="/home/cc/clio/libCacheSim/data/alibaba/ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/alibabaBlock"
trace_analyzer_dir="/home/cc/clio/libCacheSim/_build/bin/traceAnalyzer"
scripts_dir="/home/cc/clio/libCacheSim/scripts/traceAnalysis"

# Save the current working directory
orig_dir=$(pwd)

# Loop through each file in the directory that starts with 'io_traces'
for file in $data_dir/io_traces.ns*; do
    if [[ ! -f "$file" ]]; then
        echo "$file"
        continue
    fi

    # Extract the filename without the directory for use in commands
    filename=$(basename "$file")
    echo $filename

    if [[ -f "$scripts_dir/figure_named/$filename/done" ]]; then
        echo "Skipping $filename because done file is found in figures"
        continue
    fi

    # Print the absolute path of the file
    echo "Processing file: $file"

    # Navigate to trace_analyzer_result directory
    mkdir -p "trace_analyzer_result"
    cd "trace_analyzer_result"
    # Run the traceAnalyzer command
    "$trace_analyzer_dir" "$file" oracleGeneral --output=$filename --all
    cd ..

    # Navigate to scripts/traceAnalysis and run plotting function
    cd "$scripts_dir" 
    ./run.sh "$orig_dir/trace_analyzer_result/$filename"
    cd "$orig_dir" 
    
    # Return to the original directory for the next loop
    rm -rf "trace_analyzer_result"
done
