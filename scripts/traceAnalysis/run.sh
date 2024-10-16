#!/bin/bash 

dataname=$1

mkdir figure

echo "Running analysis for ${dataname}..."

echo "Running access pattern analysis..."
python3 access_pattern.py ${dataname}.accessRtime
python3 access_pattern.py ${dataname}.accessVtime

echo "Running request rate analysis..."
python3 req_rate.py ${dataname}.reqRate_w300
python3 size.py ${dataname}.size

echo "Running reuse analysis..."
python3 reuse.py ${dataname}.reuse
python3 popularity.py ${dataname}.popularity
# python3 requestAge.py ${dataname}.requestAge
echo "Running size heatmap analysis..."
python3 size_heatmap.py ${dataname}.sizeWindow_w300
# python3 futureReuse.py ${dataname}.access

echo "Running popularity decay analysis..."
python3 popularity_decay.py ${dataname}.popularityDecay_w300_obj
# python3 reuse_heatmap.py ${dataname}.reuseWindow_w300

# move /figure folder to /figure_named
mkdir -p figure_named
base_data_name=$(basename ${dataname})

mv figure figure_named/${base_data_name}
echo "MV figure to figure_named/${base_data_name}"
# make file named done
touch figure_named/${base_data_name}/done
rm -rf figure