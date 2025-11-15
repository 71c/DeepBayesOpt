import glob
import os
import re
import argparse
import subprocess
from datetime import datetime, timedelta


def get_training_stats(logs_dir):
    # Regex to capture the seconds value
    pattern_training_time = re.compile(r'Training took ([0-9.]+) seconds')
    pattern_loss = re.compile(r'Best score decreased from [0-9.]+ to ([0-9]+\.[0-9]+)')
    pattern_epoch = re.compile(r'^Epoch ([0-9]+)$')
    pattern_total_epochs = re.compile(r'--epochs ([0-9]+)')

    # Collect all times
    times = {}
    losses = {}
    in_progress = {}

    for filepath in glob.glob(os.path.join(logs_dir, 'nn*.out')):
        this_losses = []
        current_epoch = 0
        total_epochs = None
        training_completed = False

        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')

            # Find total epochs from command line
            for line in lines[:100]:  # Check first 10 lines for command
                match = pattern_total_epochs.search(line)
                if match:
                    total_epochs = int(match.group(1))
                    break

            # Process all lines
            for line in lines:
                match = pattern_loss.search(line)
                if match:
                    this_losses.append(float(match.group(1)))

                match = pattern_epoch.search(line)
                if match:
                    current_epoch = int(match.group(1))

                match = pattern_training_time.search(line)
                if match:
                    times[filepath] = float(match.group(1))
                    training_completed = True
                    break

        if this_losses:
            # Get the best loss
            best_loss = min(this_losses)
            losses[filepath] = best_loss

        # Track in-progress jobs
        if not training_completed and current_epoch > 0 and total_epochs is not None:
            in_progress[filepath] = {
                'current_epoch': current_epoch,
                'total_epochs': total_epochs,
                'best_loss': min(this_losses) if this_losses else None
            }

    return times, losses, in_progress


def extract_job_info(filepath):
    """Extract SLURM job ID and array info from filename like nn0_j9049324-A9049324_a22.out

    Returns: tuple of (individual_job_id, array_job_id, array_index)
    """
    basename = os.path.basename(filepath)
    # Pattern: nn0_j9049324-A9049324_a22.out
    # individual job: 9049324, array job: 9049324, array index: 22
    match = re.search(r'_j([0-9]+)-A([0-9]+)_a([0-9]+)', basename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    # Fallback: just extract job ID
    match = re.search(r'_j([0-9]+)', basename)
    if match:
        return match.group(1), None, None
    return None, None, None


def get_slurm_jobs():
    """Get currently running SLURM jobs for current user"""
    try:
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-o', '%.18i %.8T %.10M'],
            capture_output=True,
            text=True,
            timeout=10
        )
        jobs = {}
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 3:
                full_job_id = parts[0]  # e.g., "9049324_19" or "9049324"
                state = parts[1]
                time_str = parts[2]

                # Store both the full ID and the base ID
                base_job_id = full_job_id.split('_')[0]
                jobs[full_job_id] = {'state': state, 'time': time_str, 'full_id': full_job_id}

                # Also store by base ID if we haven't seen this base before or if this is more specific
                if base_job_id not in jobs or '_' in full_job_id:
                    jobs[base_job_id] = {'state': state, 'time': time_str, 'full_id': full_job_id}

        return jobs
    except Exception as e:
        print(f"Warning: Could not get SLURM job info: {e}")
        return {}


def parse_slurm_time(time_str):
    """Parse SLURM time format like '7:11:16' or '1-2:30:45' to seconds"""
    if '-' in time_str:
        days, rest = time_str.split('-')
        days = int(days)
    else:
        days = 0
        rest = time_str

    parts = rest.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = map(int, parts)
    else:
        return 0

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


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

    times, losses, in_progress = get_training_stats(args.logs_dir)

    # Get SLURM job information
    slurm_jobs = get_slurm_jobs()

    if times:
        argmin_time, min_time = min(times.items(), key=lambda x: x[1])
        argmax_time, max_time = max(times.items(), key=lambda x: x[1])
        average_time = sum(times.values()) / len(times)

        print("Completed Jobs - Times (d-h:m:s):")
        print(f" min: {format_time(min_time)} [{argmin_time}]")
        print(f" max: {format_time(max_time)} [{argmax_time}]")
        print(f"mean: {format_time(average_time)}")

    if losses:
        print("\nCompleted Jobs - Losses:")
        for filepath, loss in sorted(losses.items(), key=lambda x: x[1]):
            print(f"{filepath}: {loss:.6f}")

    if in_progress:
        print(f"\n{'='*100}")
        print(f"In-Progress Jobs ({len(in_progress)} total):")
        print(f"{'='*100}")

        for filepath in sorted(in_progress.keys()):
            info = in_progress[filepath]
            individual_job_id, array_job_id, array_index = extract_job_info(filepath)

            current_epoch = info['current_epoch']
            total_epochs = info['total_epochs']
            percent_complete = (current_epoch / total_epochs) * 100

            print(f"\nFile: {os.path.basename(filepath)}")

            # Try to find the job in SLURM queue
            slurm_info = None
            matched_job_id = None

            # Try different matching strategies
            if individual_job_id and individual_job_id in slurm_jobs:
                slurm_info = slurm_jobs[individual_job_id]
                matched_job_id = individual_job_id
            elif array_job_id and array_index:
                # Try array job format: 9049324_22
                array_format_id = f"{array_job_id}_{array_index}"
                if array_format_id in slurm_jobs:
                    slurm_info = slurm_jobs[array_format_id]
                    matched_job_id = array_format_id
                elif array_job_id in slurm_jobs:
                    slurm_info = slurm_jobs[array_job_id]
                    matched_job_id = array_job_id

            print(f"  Job ID: {individual_job_id} (Array: {array_job_id}_{array_index})")

            # Check if job is in SLURM queue
            if slurm_info:
                print(f"  SLURM Status: {slurm_info['state']} (matched as {matched_job_id})")
                print(f"  SLURM Time: {slurm_info['time']}")

                # Calculate time estimates
                elapsed_seconds = parse_slurm_time(slurm_info['time'])
                if current_epoch > 0:
                    seconds_per_epoch = elapsed_seconds / current_epoch
                    remaining_epochs = total_epochs - current_epoch
                    estimated_remaining = seconds_per_epoch * remaining_epochs

                    print(f"  Progress: Epoch {current_epoch}/{total_epochs} ({percent_complete:.1f}%)")
                    print(f"  Elapsed: {format_time(elapsed_seconds)}")
                    print(f"  Est. remaining: {format_time(estimated_remaining)}")
                    print(f"  Time per epoch: {format_time(seconds_per_epoch)}")
                else:
                    print(f"  Progress: Epoch {current_epoch}/{total_epochs} ({percent_complete:.1f}%)")
            else:
                print(f"  SLURM Status: NOT FOUND IN QUEUE (may have finished or failed)")
                print(f"  Progress: Epoch {current_epoch}/{total_epochs} ({percent_complete:.1f}%)")

            if info['best_loss'] is not None:
                print(f"  Best loss so far: {info['best_loss']:.6f}")

        print(f"\n{'='*100}")
        print(f"\nAll SLURM jobs in queue for current user:")
        print(f"{'='*100}")
        # Show unique jobs (avoid showing both base and array task IDs)
        seen = set()
        for job_id, job_info in sorted(slurm_jobs.items(), key=lambda x: (x[0].split('_')[0], x[0])):
            full_id = job_info['full_id']
            if full_id not in seen:
                seen.add(full_id)
                print(f"Job {full_id}: {job_info['state']} (running for {job_info['time']})")
