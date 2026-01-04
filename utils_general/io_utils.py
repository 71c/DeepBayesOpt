import json
import os
import random
import sys
import traceback


def save_json(data, fname, **kwargs):
    already_exists = os.path.exists(fname)
    r = random.randint(0, 1_000_000_000)
    save_fname = fname + f'{r}.tmp' if already_exists else fname
    try:
        # Ensure the directory exists
        dirname = os.path.dirname(fname)
        if dirname != '':
            os.makedirs(dirname, exist_ok=True)

        # Write data to the (possibly temporary) file
        with open(save_fname, 'w') as json_file:
            json.dump(data, json_file, **kwargs)

        if already_exists:
            # Replace the original file with the temporary file
            os.replace(save_fname, fname)
    except Exception as e:
        if os.path.exists(save_fname):
            # Remove the written file if an error occurs
            os.remove(save_fname)
        raise e


def load_json(fname, **kwargs):
    with open(fname, 'r') as json_file:
        try:
            return json.load(json_file, **kwargs)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from file {fname}.\\Error message:\n",
                  file=sys.stderr)
            traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
            exit(66)
