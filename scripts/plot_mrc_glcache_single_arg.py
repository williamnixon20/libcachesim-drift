import os
import sys
import itertools
from collections import defaultdict
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Dict, Tuple, Union, Literal
import subprocess
import logging
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.plot_utils import *
from utils.trace_utils import extract_dataname
from utils.str_utils import conv_size_str_to_int, find_unit_of_cache_size
from utils.setup_utils import setup, CACHESIM_PATH

logger = logging.getLogger("plot_mrc_time")


REGEX = r"(?P<hour>\d+\.\d+) hour: (?P<nreq>\d+) requests, miss ratio (?P<miss_ratio>\d+\.\d+), interval miss ratio (?P<interval_miss_ratio>\d+\.\d+)"


# Function to check if an experiment result exists in the CSV
def load_existing_results(csv_file: str) -> Dict[str, Tuple[List[float], List[float]]]:
    if not os.path.exists(csv_file):
        return {}

    existing_results = defaultdict(lambda: ([], []))
    df = pd.read_csv(csv_file)
    for algo, time, miss_ratio in zip(df["Algorithm"], df["Time"], df["Miss Ratio"]):
        existing_results[algo][0].append(time)
        existing_results[algo][1].append(miss_ratio)
    return existing_results


def run_cachesim_time_custom(
    datapath: str,
    cache_size: Union[int, str],
    ignore_obj_size: bool = True,
    miss_ratio_type: Literal["accu", "interval"] = "interval",
    report_interval: int = 3600,
    byte_miss_ratio: bool = False,  # not used
    trace_format: str = "oracleGeneral",
    trace_format_args: str = "",
    retrain_duration: int = 9999999,
    should_save:bool = False,
    should_load:bool = False,
    model_file:str = "",
    is_matchmaker:bool = False,
    is_aue:bool = False,
    label:str = "",
    algo: str = "gl-cache",
    warmup_sec: int = 86400,
) -> Tuple[List[float], List[float]]:
    """run the cachesim on the given trace to obtain how miss ratio change over time
    Args:
        datapath: the path to the trace
        algo: the algo to run
        cache_size: the cache size to run
        ignore_obj_size: whether to ignore the object size, default: True
        miss_ratio_type: the type of miss ratio, default: interval
        interval: the interval to report the miss ratio, default: 3600
        byte_miss_ratio: whether to report the miss ratio in byte, default: False
        trace_format: the trace format, default: oracleGeneral
        num_thread: the number of threads to run, default: -1 (use all the cores)
    """

    ts_list, mrc_list = [], []

    run_args = [
        CACHESIM_PATH,
        datapath,
        trace_format,
        algo,
        str(cache_size),
        "--report-interval",
        str(report_interval),
        "--ignore-obj-size",
        "1" if ignore_obj_size else "0",
        "--dump-model",
        str(should_save).lower(),
        "--load-model",
        str(should_load).lower(),
        # "--model-file",
        # str(model_file),
        "--matchmaker",
        str(is_matchmaker).lower(),
        "--aue",
        str(is_aue).lower(),
        "--label",
        str(label).lower(),
        "--warmup-sec",
        str(warmup_sec),
    ]

    # if len(trace_format_args) > 0:
    #     run_args.append("--trace-type-params")
    #     run_args.append(trace_format_args)
    run_args.append("--retrain-intvl")
    run_args.append(str(retrain_duration))
    

    logger.debug('running "{}"'.format(" ".join(run_args)))

    p = subprocess.run(run_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        logger.warning("cachesim may have crashed with segfault")
        # Make an error file, then print the error log
        with open("error.txt", "w") as f:
            f.write(p.stderr.decode("utf-8"))
    try:
        stdout_str = p.stdout.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        print("ERROR IN DECODING")
        print(p.stdout)
        
    for line in stdout_str.split("\n"):
        # logger.debug("cachesim log: " + line + " SIZE CACHE" + str(cache_size))

        # python3 plot_mrc_time.py --tracepath ../data/cloudPhysicsIO.oracleGeneral.bin --trace-format oracleGeneral --algos=fifo,lru,lecar,s3fifo --report-interval=30 --miss-ratio-type="accu"
        # python3 plot_mrc_time.py --tracepath ../data/twitter_cluster52.csv --trace-format csv --trace-format-params="time-col=1, obj-id-col=2, obj-size-col=3, delimiter=,," --trace-format csv --algos=fifo,lru --report-interval=100 --miss-ratio-type="accu" --verbose 
        if "[INFO]" in line[:16]:
            m = re.search(REGEX, line)
            if not m:
                continue
            ts_list.append(float((m.group("hour"))))
            if miss_ratio_type == "accu":
                mrc_list.append(float(m.group("miss_ratio")))
            elif miss_ratio_type == "interval":
                mrc_list.append(float(m.group("interval_miss_ratio")))
            else:
                raise Exception("Unknown miss ratio type {}".format(miss_ratio_type))
        else:
            ...
        if line.startswith("result"):
            logger.debug(line)
    # print("FINISH TIME", mrc_list, ts_list)
    return ts_list, mrc_list

def plot_mrc_time(mrc_dict, name="mrc"):
    linestyles = itertools.cycle(["-", "--", "-.", ":"])
    colors = itertools.cycle(
        [
            "navy",
            "darkorange",
            "tab:green",
            "cornflowerblue",
        ]
    )
    MARKERS = itertools.cycle(Line2D.markers.keys())

    data = []
    for algo, (ts, mrc) in mrc_dict.items():
        ts = np.array(ts) / ts[-1]
        plt.plot(
            ts,
            mrc,
            linewidth=4,
            color=next(colors),
            linestyle=next(linestyles),
            label=algo,
        )
        for t, miss in zip(ts, mrc):
            data.append([algo, t, miss])  # Fixed this line

    plt.xlabel("Time")
    plt.ylabel("Miss Ratio")
    plt.title("Name: {}".format(name))
    # Place the legend outside the plot
    plt.legend(ncol=2, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

    plt.grid(axis="y", linestyle="--")
    plt.savefig("{}.pdf".format(name), bbox_inches="tight")
    plt.show()
    plt.clf()
    print("plot is saved to {}.pdf".format(name))
    
    # Save the data to a CSV file
    df = pd.DataFrame(data, columns=["Algorithm", "Time", "Miss Ratio"])
    df.to_csv("mrc_list.csv", index=False)
    print("Data saved to mrc_list.csv")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="plot miss ratio over time for different algorithms")
    parser.add_argument("--tracepath", type=str, required=True)
    parser.add_argument("--algos", type=str, default="lru,arc,lhd,tinylfu,s3fifo,sieve")
    parser.add_argument("--size", type=str, default="0.1")
    parser.add_argument("--miss-ratio-type", type=str, default="accu")
    parser.add_argument("--report-interval", type=int, default=3600)
    parser.add_argument("--trace-format", type=str, default="oracleGeneral")
    parser.add_argument("--ignore-obj-size", action="store_true", default=False)
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG)

    csv_file = "mrc_list.csv"
    existing_results = load_existing_results(csv_file)
    mrc_dict = defaultdict(lambda: ([], []))
    
    tracepath = args.tracepath
    size = args.size

    label_variants = {
        "every-day": {"algo": "gl-cache", "retrain_duration": 86400, "label": "every-day", "warmup_sec": 86400},
        # "no-retrain": {"algo": "gl-cache", "retrain_duration": 9999999, "label": "no-retrain"},
        "aue": {"algo": "gl-cache", "retrain_duration": 86400, "should_save": True, "is_aue": True, "label": "aue", "warmup_sec": 86400},
        "matchmaker": {"algo": "gl-cache", "retrain_duration": 86400, "should_save": True, "is_matchmaker": True, "label": "matchmaker", "warmup_sec": 86400},
    }
    for label, params in label_variants.items():
        algo = params["algo"]
        algo_label = f"{algo}-{label}"
        if algo_label in existing_results:
            mrc_dict[algo_label] = existing_results[algo_label]
            print(f"Skipping {algo_label} (already computed)")
        else:
            print("Running", algo)
            ts, mrc = run_cachesim_time_custom(
                tracepath, size, args.ignore_obj_size, args.miss_ratio_type, args.report_interval,
                **params
            )
            
            mrc_dict[algo_label] = (ts, mrc)
            plot_mrc_time(mrc_dict, os.path.splitext(os.path.basename(tracepath))[0])

    plot_mrc_time(mrc_dict, os.path.splitext(os.path.basename(tracepath))[0])

if __name__ == "__main__":
    main()

## /home/cc/libcachesim-private/_build/bin/cachesim /home/cc/libcachesim-private/data/ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/alibabaBlock/io_traces.ns370.oracleGeneral.zst oracleGeneral gl-cache 0.1 --report-interval 3600 --ignore-obj-size 0 --dump-model false --load-model false --matchmaker false --aue false --label every-day --warmup-sec 86400 --retrain-intvl 86400
## /home/cc/libcachesim-private/_build/bin/cachesim /home/cc/libcachesim-private/data/ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/alibabaBlock/io_traces.ns370.oracleGeneral.zst oracleGeneral gl-cache 0.1 --report-interval 3600 --ignore-obj-size 0 --dump-model false --load-model false --matchmaker false --aue true --label aue --warmup-sec 86400 --retrain-intvl 86400
## /home/cc/libcachesim-private/_build/bin/cachesim /home/cc/libcachesim-private/data/ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/alibabaBlock/io_traces.ns370.oracleGeneral.zst oracleGeneral gl-cache 0.1 --report-interval 3600 --ignore-obj-size 0 --dump-model false --load-model false --matchmaker true --aue false --label aue --warmup-sec 86400 --retrain-intvl 86400