#!/bin/bash

# Define directories
data_dir="/home/cc/clio/libCacheSim/data/alibaba/ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/alibabaBlock"

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

    # Check if file size is larger than 20 MB
    if [[ $(stat -c%s "$file") -gt 200000000 ]]; then
        echo "Skipping $file because file size is larger than 20 MB"
        return
    fi

    # Check if processing is already done
    if [[ -f "result__/$filename/done" ]]; then
        echo "Skipping $filename because done file is found"
        return
    fi

    rm -rf result__/$filename
    mkdir -p result__/$filename/dump
    cd result__/$filename
    mkdir dump

    # Run your Python scripts (uncomment the lines you want to run)
    python3 ../../plot_mrc_glcache_multimodel.py --tracepath $file --algos=gl-cache --miss-ratio-type="accu" --verbose 
    
    touch done

    cd $orig_dir
}

export -f process_file

# Export the result__ directory to ensure it is accessible inside the process_file function
export orig_dir
export data_dir

# Loop through each file and process in parallel
find "$data_dir" -name "io_traces.ns*" | echo "Processing file name $(xargs -n 1 basename)" | xargs -n 1 -P 30 bash -c 'process_file "$@"' _
