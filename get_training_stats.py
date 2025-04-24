import glob
import os
import re
import argparse
from datetime import datetime, timedelta


def get_training_stats(logs_dir):
    # Regex to capture the seconds value
    pattern_training_time = re.compile(r'Training took ([0-9.]+) seconds')
    pattern_loss = re.compile(r'Best score decreased from [0-9.]+ to ([0-9]+\.[0-9]+)')

    # Collect all times
    times = {}
    losses = {}
    for filepath in glob.glob(os.path.join(logs_dir, 'nn*.out')):
        filename = os.path.basename(filepath)
        this_losses = []
        with open(filepath, 'r') as f:
            for line in f:
                match = pattern_loss.search(line)
                if match:
                    this_losses.append(float(match.group(1)))
                match = pattern_training_time.search(line)
                if match:
                    times[filename] = float(match.group(1))
                    break
        
        if this_losses:
            # Get the best loss
            best_loss = min(this_losses)
            losses[filename] = best_loss
    return times, losses


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

    times, losses = get_training_stats(args.logs_dir)
    if times:
        argmin_time, min_time = min(times.items(), key=lambda x: x[1])
        argmax_time, max_time = max(times.items(), key=lambda x: x[1])
        average_time = sum(times.values()) / len(times)
    
        print("Times (d-h:m:s):")
        print(f" min: {format_time(min_time)} [{argmin_time}]")
        print(f" max: {format_time(max_time)} [{argmax_time}]")
        print(f"mean: {format_time(average_time)}")
    
    if losses:
        print("\nLosses:")
        for filename, loss in sorted(losses.items(), key=lambda x: x[1]):
            print(f"{filename}: {loss:.6f}")


# python get_training_stats.py data/sweeps/100iter_8dim_maxhistory20_big_20250417_194250/logs
# python get_training_stats.py data/sweeps/100iter_8dim_maxhistory20_gittins_regularization_2_20250421_202615/logs
# python get_training_stats.py data/sweeps/100iter_8dim_maxhistory20_gittins_regularization_2_20250422_221933/logs
# python get_training_stats.py data/sweeps/100iter_8dim_maxhistory20_regularization_20250424_142837/logs
