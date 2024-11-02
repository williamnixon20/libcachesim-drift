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



def run_cachesim_time_custom(
    datapath: str,
    algo: str,
    cache_size: Union[int, str],
    ignore_obj_size: bool = True,
    miss_ratio_type: Literal["accu", "interval"] = "interval",
    report_interval: int = 3600,
    byte_miss_ratio: bool = False,  # not used
    trace_format: str = "oracleGeneral",
    trace_format_args: str = "",
    num_thread: int = -1,
    retrain_duration: int = 9999999,
    should_save:bool = False,
    should_load:bool = False,
    model_file:str = "",
    is_matchmaker:bool = False,
    label:str = ""
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
    if num_thread < 0:
        num_thread = os.cpu_count()

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
        "--num-thread",
        str(1),
        "--dump-model",
        str(should_save).lower(),
        "--load-model",
        str(should_load).lower(),
        "--model-file",
        str(model_file),
        "--matchmaker",
        str(is_matchmaker).lower(),
        "--label",
        str(label).lower()
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

    stderr_str = p.stderr.decode("utf-8")
    if stderr_str != "":
        logger.warning(stderr_str)

    stdout_str = p.stdout.decode("utf-8")
    stdout_str += stderr_str
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
    
# def plot_mrc_time(mrc_dict, name="mrc"):
#     linestyles = itertools.cycle(["-", "--", "-.", ":"])
#     linestyles = itertools.cycle(["--", "-", "--", "-.", ":"])
#     colors = itertools.cycle(
#         [
#             "navy",
#             "darkorange",
#             "tab:green",
#             "cornflowerblue",
#         ]
#     )
#     MARKERS = itertools.cycle(Line2D.markers.keys())

#     data = []
#     for algo, (ts, mrc) in mrc_dict.items():
#         ts = np.array(ts) / ts[-1]
#         plt.plot(
#             ts,
#             mrc,
#             linewidth=4,
#             color=next(colors),
#             linestyle=next(linestyles),
#             label=algo,
#         )
#         for t, miss in zip(ts, mrc):
#             data.append([algo, t, mrc])

#     plt.xlabel("Time")
#     plt.ylabel("Miss Ratio")
#     legend = plt.legend(ncol=2, loc="upper left", frameon=False)
#     frame = legend.get_frame()
#     frame.set_facecolor("0.9")
#     frame.set_edgecolor("0.9")
#     plt.grid(axis="y", linestyle="--")
#     plt.savefig("{}.pdf".format(name), bbox_inches="tight")
#     plt.show()
#     plt.clf()
#     print("plot is saved to {}.pdf".format(name))
#     from pandas import pd
#     df = pd.DataFrame(data, columns=["Algorithm", "Time", "Miss Ratio"])
    
#     # Save the DataFrame to a CSV file
#     df.to_csv("mrc_list.csv", index=False)


def run():
    import glob

    algos = "lru,arc,lhd,tinylfu,s3fifo,sieve"
    cache_sizes = "0"
    for i in range(1, 100, 2):
        cache_sizes += str(i / 100.0) + ","
    cache_sizes = cache_sizes[:-1]

    for tracepath in glob.glob("/disk/data/*.zst"):
        dataname = tracepath.split("/")[-1].split(".")[0]
        # mrc_dict = run_cachesim_time(tracepath, algos, cache_sizes)
        # print("MRC DICT", mrc_dict)
        mrc_dict_no_retrain = run_cachesim_time_custom(tracepath, algos, cache_sizes)
        print("MRC DICT NO RETRAIN", mrc_dict_no_retrain)
        # join the 2 dict
        for algo in mrc_dict_no_retrain:
            mrc_dict[algo + "-no-retrain"] += mrc_dict_no_retrain[algo]
            
        plot_mrc_time(mrc_dict, dataname)


if __name__ == "__main__":
    default_args = {
        "algos": "lru,arc,lhd,tinylfu,s3fifo,sieve",
        "size": 0.1,
        "miss_ratio_type": "accu",
        "report_interval": "3600",
    }

    import argparse

    p = argparse.ArgumentParser(
        description="plot miss ratio over size for different algorithms, example: \n"
        "python3 {} ".format(sys.argv[0]) + "--tracepath ../data/twitter_cluster52.csv "
        "--trace-format csv "
        '--trace-format-params="time-col=1,obj-id-col=2,obj-size-col=3,delimiter=,,obj-id-is-num=1" '
        "--algos=fifo,lru,lecar,s3fifo "
        "--report-interval 120"
    )
    p.add_argument("--tracepath", type=str, required=True)
    p.add_argument("--algos", type=str, default=default_args["algos"])
    p.add_argument("--size", type=str, default=default_args["size"])
    p.add_argument(
        "--miss-ratio-type", type=str, default=default_args["miss_ratio_type"]
    )
    p.add_argument(
        "--report-interval", type=int, default=default_args["report_interval"]
    )
    p.add_argument("--trace-format", type=str, default="oracleGeneral")
    p.add_argument(
        "--trace-format-params", type=str, default="", help="used by csv trace"
    )
    p.add_argument("--ignore-obj-size", action="store_true", default=False)
    p.add_argument("--byte-miss-ratio", action="store_true", default=False)
    p.add_argument("--num-thread", type=int, default=-1)
    p.add_argument("--name", type=str, default="")
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--test", action="store_true", default=False)
    ap = p.parse_args()

    if ap.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if ap.test:
        run()
        sys.exit(0)

    mrc_dict = defaultdict(list)
    dataname = extract_dataname(ap.tracepath)

    for algo in ap.algos.split(","):
        mrc_dict[algo+'-every-day'] = run_cachesim_time_custom(
            ap.tracepath,
            algo,
            ap.size,
            ap.ignore_obj_size,
            ap.miss_ratio_type,
            ap.report_interval,
            ap.byte_miss_ratio,
            ap.trace_format,
            ap.trace_format_params,
            ap.num_thread,
            retrain_duration=86400*2,
            label="every-day"
        )
        
        mrc_dict[algo+"-no-retrain"] = run_cachesim_time_custom(
            ap.tracepath,
            algo,
            ap.size,
            ap.ignore_obj_size,
            ap.miss_ratio_type,
            ap.report_interval,
            ap.byte_miss_ratio,
            ap.trace_format,
            ap.trace_format_params,
            ap.num_thread,
            # effectively never retrains
            retrain_duration=9999999,
            label="no-retrain"
        )
        mrc_dict[algo+'-retrain-pick-concept-best'] = run_cachesim_time_custom(
            ap.tracepath,
            algo,
            ap.size,
            ap.ignore_obj_size,
            ap.miss_ratio_type,
            ap.report_interval,
            ap.byte_miss_ratio,
            ap.trace_format,
            ap.trace_format_params,
            ap.num_thread,
            retrain_duration=86400*2,
            should_save=True,
            is_matchmaker=True,
            label="matchmake-concept"
        )
        
        ap.algos = "gl-cache-every-day,gl-cache-no-retrain, gl-cache-retrain-pick-concept-best"
        

    if len(mrc_dict) == 0:
        logger.error("fail to compute mrc")
        sys.exit(1)

    plot_mrc_time(mrc_dict, "{}_{}".format(dataname, ap.size))
