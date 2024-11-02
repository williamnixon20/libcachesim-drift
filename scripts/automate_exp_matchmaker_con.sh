#!/bin/bash

# Define directories
# data_dir="/home/cc/clio/libCacheSim/data/alibaba/ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/alibabaBlock"
data_dir="/home/cc/libcachesim-private/data/ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/alibabaBlock"

# Save the current working directory
orig_dir=$(pwd)

# Function to process each file
process_file() {
    local file="$1"
    local filename=$(basename "$file")
    # concat filename with datadir
    file="$data_dir/$filename"

    # Check if file exists
    if [[ ! -f "$file" ]]; then
        echo "$file does not exist"
        return
    fi

    # Check if file size is larger than 200 MB
    if [[ $(stat -c%s "$file") -gt 1000000000 ]]; then
        echo "Skipping $file because file size is larger than 20 MB"
        return
    fi

    base_result_dir="result_matchmaker"

    # Check if processing is already done
    if [[ -f "$base_result_dir/$filename/done" ]]; then
        echo "Skipping $filename because done file is found"
        return
    fi

    rm -rf $base_result_dir/$filename
    mkdir -p $base_result_dir/$filename/dump
    cd $base_result_dir/$filename
    mkdir dump

    # Run your Python scripts (uncomment the lines you want to run)
    python3 ../../plot_mrc_glcache_matchmaker_concept.py --tracepath $file --algos=gl-cache --miss-ratio-type="accu" --verbose 
    
    touch done

    cd $orig_dir
}

export -f process_file

# Export the result__ directory to ensure it is accessible inside the process_file function
export orig_dir
export data_dir

# # Loop through each file and process in parallel
# find "$data_dir" -name "*oracleGeneral*" | echo "Processing file name $(xargs -n 1 basename)" | xargs -n 1 -P 30 bash -c 'process_file "$@"' _
# find "$data_dir" -name "*oracleGeneral*" | xargs -n 40 -I {} bash -c 'echo "Processing file name $(basename "{}")"; process_file "$@"' _ {}
find "$data_dir" -name "*oracleGeneral*" | xargs -P 40 -I {} bash -c 'echo "Processing file name $(basename "{}")"; process_file "$@"' _ {}

## /home/cc/libcachesim-private/_build/bin/cachesim /home/cc/libcachesim-private/data/ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/alibabaBlock/io_traces.ns251.oracleGeneral.zst oracleGeneral gl-cache 0.1 --report-interval 3600 --ignore-obj-size 0 --num-thread 48 --dump-model true --load-model false --matchmaker true --label matchmaki --retrain-intvl 172800