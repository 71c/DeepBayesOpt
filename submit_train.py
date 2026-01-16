import argparse
from functools import cache

from utils_general.experiments.submit_train_utils import SubmitTrainUtils
from datasets.utils import get_cmd_options_sample_dataset
from single_train_baseline import get_dataset_hash_for_transfer_bo_baselines, transfer_bo_baseline_is_trained
from utils_general.utils import dict_to_cmd_args, get_arg_names

from dataset_factory import add_common_acquisition_dataset_args, add_lamda_args, create_train_test_acquisition_datasets_from_args
from utils_train.model_save_utils import ACQF_NET_SAVING


@cache
def _get_common_acquisition_dataset_arg_names():
    parser = argparse.ArgumentParser()
    add_common_acquisition_dataset_args(parser)
    return set(get_arg_names(parser))


@cache
def _get_lamda_arg_names():
    parser = argparse.ArgumentParser()
    add_lamda_args(parser)
    return set(get_arg_names(parser))


class AcquisitionFunctionSubmitTrainUtils(SubmitTrainUtils):
    special_args = _get_lamda_arg_names()

    @classmethod
    def create_datasets_func(
        cls,
        args: argparse.Namespace,
        check_cached: bool = False,
        load_dataset: bool = True
    ):
        return create_train_test_acquisition_datasets_from_args(
            args, check_cached=check_cached, load_dataset=load_dataset)

    def get_cmd_opts_dataset(self, options: dict) -> dict:
        cmd_opts_sample_dataset = get_cmd_options_sample_dataset(options)
        # Acquisition dataset arguments (not included in sample dataset)
        acquisition_arg_names = _get_common_acquisition_dataset_arg_names()
        acquisition_arg_names.update(self.special_args)
        tmp = {'train_n_candidates', 'test_n_candidates'}
        cmd_opts_acquisition_dataset = {
            k: options.get(k if k not in tmp else 'n_candidates')
            for k in acquisition_arg_names
        }
        # Combine sample and acquisition dataset options
        return {**cmd_opts_sample_dataset, **cmd_opts_acquisition_dataset}

    def determine_whether_trained(self, options: dict, cmd_opts_nn: dict) -> bool:
        transfer_bo_method = options.get('transfer_bo_method', None)
        if transfer_bo_method is not None:
            dataset_hash = get_dataset_hash_for_transfer_bo_baselines(options)
            return transfer_bo_baseline_is_trained(transfer_bo_method, dataset_hash)
        else:
            return super().determine_whether_trained(options, cmd_opts_nn)

    def get_train_cmd_options(self, options, cmd_opts_dataset, cmd_args_dataset):
        transfer_bo_method = options.get('transfer_bo_method', None)
        if transfer_bo_method is not None:
            # Baseline transfer BO method
            cmd_opts_dataset_no_special = {
                k: v for k, v in cmd_opts_dataset.items()
                if k not in self.special_args
            }

            cmd_opts_nn = {
                'transfer_bo_method': options.get('transfer_bo_method'),
                **cmd_opts_dataset_no_special
            }
            
            cmd_nn_train = " ".join(["python single_train_baseline.py",
                                    *dict_to_cmd_args(cmd_opts_nn)])
        else:
            cmd_nn_train, cmd_opts_nn = super().get_train_cmd_options(
                options, cmd_opts_dataset, cmd_args_dataset)
        
        return cmd_nn_train, cmd_opts_nn


AF_TRAIN_SUBMIT_UTILS = AcquisitionFunctionSubmitTrainUtils(ACQF_NET_SAVING)


def main():
    AF_TRAIN_SUBMIT_UTILS.submit_jobs()


if __name__ == "__main__":
    main()
