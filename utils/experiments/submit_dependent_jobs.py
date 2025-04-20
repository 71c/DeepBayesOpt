from sys import stderr
from typing import Optional
import subprocess
import os
import argparse
from datetime import datetime
import math

from utils.constants import JOB_ARRAY_SUB_PATH, SWEEPS_DIR
from utils.experiments.experiment_config_utils import CONFIG_DIR
from utils.utils import dict_to_cmd_args, save_json

# scontrol show config | grep -E 'MaxArraySize'
# it says 1001 but 1001 doesn't work, so it looks like 1000 is the max
MAX_ARRAY_SIZE = 1000


def add_slurm_args(parser):
    parser.add_argument(
        '--sweep_name',
        type=str,
        default='test',
        help='Name of the sweep.'
    )
    parser.add_argument(
        '--gres',
        type=str,
        help='GPU resource specification for Slurm. e.g., "gpu:a100:1" or "gpu:1". '
              'Default is "gpu:a100:1".',
        default="gpu:a100:1"
    )
    parser.add_argument(
        '--mail',
        type=str,
        help='email address to send Slurm notifications to. '
              'If not specified, no notifications are sent.'
    )
    parser.add_argument(
        '--no_submit',
        action='store_true',
        help='If specified, do not submit jobs, but only save dependencies.json so '
                'that you can see what would be submitted.'
    )


def submit_jobs_sweep_from_args(jobs_spec, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(SWEEPS_DIR, f"{args.sweep_name}_{timestamp}")
    _submit_dependent_jobs(
        sweep_dir=sweep_dir,
        jobs_spec=jobs_spec,
        args=args,
        gres=args.gres,
        mail=args.mail,
        no_submit=args.no_submit
    )
 

def _split_jobs_spec(jobs_spec):
    """Split jobs_spec into chunks of size MAX_ARRAY_SIZE."""
    ret = {}
    for job_name, job_spec in jobs_spec.items():
        n_commands = len(job_spec["commands"])
        if n_commands > MAX_ARRAY_SIZE:
            n_chunks = math.ceil(n_commands / MAX_ARRAY_SIZE)
            for i in range(n_chunks):
                chunk_name = f"{job_name}_{i+1}"
                chunk_spec = job_spec.copy()
                chunk_spec["commands"] = job_spec["commands"][i*MAX_ARRAY_SIZE:(i+1)*MAX_ARRAY_SIZE]
                ret[chunk_name] = chunk_spec
        else:
            ret[job_name] = job_spec
    return ret


def _submit_dependent_jobs(
        sweep_dir: str,
        jobs_spec: dict,
        args: argparse.Namespace,
        gres: str = "gpu:a100:1", # e.g. "gpu:a100:1" or "gpu:1"
        mail:Optional[str]=None,
        no_submit:bool=False
    ):
    # Split jobs_spec into chunks of size MAX_ARRAY_SIZE
    jobs_spec = _split_jobs_spec(jobs_spec)

    save_json(jobs_spec, os.path.join(CONFIG_DIR, "dependencies.json"), indent=4)

    # Don't create unnecessary directories if there are no jobs to submit
    # (i.e. if jobs_spec is empty)
    # Also don't submit if no_submit is True
    if not jobs_spec or no_submit:
        return

    logs_dir = os.path.join(sweep_dir, "logs")
    config_dir = os.path.join(sweep_dir, "config")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    save_json(vars(args), os.path.join(sweep_dir, "args.json"))

    save_json(jobs_spec, os.path.join(sweep_dir, "dependencies.json"), indent=4)

    job_ids = {}

    def submit_job(job_name):
        if job_name in job_ids:
            return job_ids[job_name]
        
        job_spec = jobs_spec[job_name]
        # Submit all the parent jobs if we haven't already,
        # and get their job ids
        prerequisites = job_spec.get("prerequisites", [])
        dependency_job_ids = []
        for prerequisite in prerequisites:
            d_job_name = prerequisite["job_name"]
            if d_job_name not in jobs_spec:
                raise ValueError(
                    f"invalid format -- job id {d_job_name} is not in jobs_spec")
            
            dependency_job_id = submit_job(d_job_name)
            
            if "index" in prerequisite:
                d_index = prerequisite["index"]
                n_jobs_dependency = len(jobs_spec[d_job_name]["commands"])
                if not (1 <= d_index <= n_jobs_dependency):
                    raise ValueError(
                        f"Index {d_index} out of range 1...{n_jobs_dependency} "
                        f"for dependency {d_job_name}")
                dependency_job_ids.append(f"{dependency_job_id}_{d_index}")
            else:
                dependency_job_ids.append(f"{dependency_job_id}")

        # Create file with a list of all the commands
        commands_list_fname = f"{job_name}_commands.txt"
        commands_list_fpath = os.path.join(config_dir, commands_list_fname)
        with open(commands_list_fpath, "w") as f:
            f.writelines([cmd + "\n" for cmd in job_spec["commands"]])
        
        n_commands = len(job_spec["commands"])
        sbatch_args_dict = {
            "job-name": job_name,
            "output": os.path.join(logs_dir, f"{job_name}_j%j-A%A_a%a.out"),
            "error": os.path.join(logs_dir, f"{job_name}_j%j-A%A_a%a.err"),
            "array": f"1-{n_commands}",
            "mem": "64gb" # server memory requested (per node)
        }
        sbatch_args_dict['time'] = job_spec.get("time", "48:00:00")
        if mail is not None:
            sbatch_args_dict['mail-type'] = 'ALL'
            sbatch_args_dict['mail-user'] = mail
        if job_spec["gpu"]:
            sbatch_args_dict['partition'] = 'gpu'
            sbatch_args_dict['gres'] = job_spec.get("gres", gres)
            # sbatch_args_dict['mem'] = "20gb"
        else:
            # sbatch_args_dict['partition'] = 'frazier'
            sbatch_args_dict['mem'] = '8gb'
        if dependency_job_ids:
            dependency_flag = 'afterok:' + ':'.join(dependency_job_ids)
            sbatch_args_dict['dependency'] = dependency_flag
        
        sbatch_args = dict_to_cmd_args(sbatch_args_dict, equals=True)
        args = ["sbatch"] + sbatch_args + [
            JOB_ARRAY_SUB_PATH, commands_list_fpath, sweep_dir]
        print(" ".join(args))
        result = subprocess.run(
            args,
            capture_output=True,
            text=True
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            cmd = " ".join(args)
            stderr = result.stderr
            raise RuntimeError(f"Failed to submit job\n {cmd}\nOutput:\n{output}\nError:\n{stderr}")
        job_id = output.split()[-1]
        job_ids[job_name] = job_id
        return job_id
    
    for j_name in jobs_spec:
        submit_job(j_name)
    save_json(job_ids, os.path.join(sweep_dir, "job_ids.json"))
