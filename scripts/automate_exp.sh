#!/bin/bash

# Define directories
data_dir="/home/cc/clio/libCacheSim/data/alibaba/ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/alibabaBlock"

# Save the current working directory
orig_dir=$(pwd)

# Loop through each file in the directory that starts with 'io_traces'
for file in $data_dir/io_traces.ns*; do
    if [[ ! -f "$file" ]]; then
        echo "$file"
        continue
    fi
    # ## if size of file is larger than 10 mb, skip
    # if [[ $(stat -c%s "$file") -gt 1000000000 ]]; then
    #     echo "Skipping $file because file size is larger than 10 mb"
    #     continue
    # fi

    # Extract the filename without the directory for use in commands
    filename=$(basename "$file")
    # echo file being processed and filesize
    echo $filename
    echo $(stat -c%s "$file")

    if [[ -f "result_/$filename/done" ]]; then
        echo "Skipping $filename because done file is found exp results"
        continue
    fi

    mkdir -p result_/$filename
    cd result_/$filename
    echo "CURR DIR $(pwd)"

    # python3 ../../plot_mrc_time.py --tracepath $file --algos=fifo,lru,lfu,arc,lecar,lhd,tinylfu,s3fifo,sieve --miss-ratio-type="accu" --verbose 
    # python3 ../../plot_mrc_size.py --tracepath $file --algos=fifo,lru,lfu,arc,lecar,lhd,tinylfu,s3fifo,sieve --sizes=0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4
    # python3 ../../plot_mrc_size.py --tracepath $file --algos=glcache --sizes=0.001,0.005,0.01,0.05,0.1,0.2,0.3
    # python3 ../../plot_mrc_time.py --tracepath $file --algos=gl-cache,lhd,s3fifo --miss-ratio-type="accu" --verbose 
    python3 ../../plot_mrc_retrain_glcache.py --tracepath $file --algos=gl-cache --miss-ratio-type="accu" --verbose 
    touch done

    cd $orig_dir
done
