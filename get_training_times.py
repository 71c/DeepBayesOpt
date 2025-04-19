import glob
import os
import re
import argparse
from datetime import datetime, timedelta


def get_training_times(logs_dir):
    # Regex to capture the seconds value
    pattern = re.compile(r'Training took ([0-9.]+) seconds')

    # Collect all times
    times = []
    for filepath in glob.glob(os.path.join(logs_dir, 'nn*.out')):
        with open(filepath, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    times.append(float(match.group(1)))
                    break

    return times


# https://stackoverflow.com/a/4048773
def format_time(seconds):
    d = datetime(1,1,1) + timedelta(seconds=seconds)
    ret = "%d:%d:%d" % (d.hour, d.minute, d.second)
    days = d.day - 1
    if days > 0:
        ret = "%2d-%s" % (days, ret)
    else:
        ret = "   " + ret
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get training times from .out files.')
    parser.add_argument('logs_dir', type=str, help='Directory containing the .out logs')
    args = parser.parse_args()

    times = get_training_times(args.logs_dir)
    if times:
        min_time = min(times)
        max_time = max(times)
        average_time = sum(times) / len(times)
    
        print("Times (d-h:m:s):")
        print(f" min: {format_time(min_time)}")
        print(f" max: {format_time(max_time)}")
        print(f"mean: {format_time(average_time)}")

# python get_training_times.py data/sweeps/100iter_8dim_maxhistory20_big_20250417_194250/logs
