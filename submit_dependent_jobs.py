from typing import Optional
import subprocess
import os

from utils import dict_to_cmd_args, save_json


CONFIG_DIR = "config"
SWEEPS_DIR = "sweeps"


def submit_dependent_jobs(
        sweep_dir: str,
        jobs_spec: dict,
        gpu_gres: str = "gpu:a100:1", # e.g. "gpu:a100:1" or "gpu:1"
        mail:Optional[str]=None
    ):
    # Don't create unnecessary directories if there are no jobs to submit
    # (i.e. if jobs_spec is empty)
    if not jobs_spec:
        return

    logs_dir = os.path.join(sweep_dir, "logs")
    config_dir = os.path.join(sweep_dir, "config")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

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
            "output": os.path.join(logs_dir, f"{job_name}_%j.out"),
            "error": os.path.join(logs_dir, f"{job_name}_%j.err"),
            "requeue": True,
            "array": f"1-{n_commands}",
            "mem": "40gb" # server memory requested (per node)
        }
        if mail is not None:
            sbatch_args_dict['mail-type'] = 'ALL'
            sbatch_args_dict['mail-user'] = mail
        if job_spec["gpu"]:
            sbatch_args_dict['partition'] = 'gpu'
            sbatch_args_dict['gres'] = gpu_gres
        else:
            sbatch_args_dict['partition'] = 'frazier'
        if dependency_job_ids:
            dependency_flag = 'afterok:' + ':'.join(dependency_job_ids)
            sbatch_args_dict['dependency'] = dependency_flag
        
        sbatch_args = dict_to_cmd_args(sbatch_args_dict, equals=True)
        args = ["sbatch"] + sbatch_args + ["job_array.sub", commands_list_fpath]
        print(" ".join(args))
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            text=True
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            cmd = " ".join(args)
            raise RuntimeError(f"Failed to submit job\n {cmd}\nOutput:\n{output}")
        job_id = output.split()[-1]
        job_ids[job_name] = job_id
        return job_id
    
    for j_name in jobs_spec:
        submit_job(j_name)
    save_json(job_ids, os.path.join(sweep_dir, "job_ids.json"))
