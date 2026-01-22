import math
from typing import Optional
from types import SimpleNamespace

from utils_general.training.single_trainer import SingleTrainer
from utils_train.model_save_utils import ACQF_NET_SAVING
from dataset_factory import create_train_test_acquisition_datasets_from_args
from utils_train.train_or_test_loop import AcquisitionFunctionTrainOrTestLoop


_BASIC_STATS = {"ei_random_search", "ei_ideal", "mse_always_predict_0"}

_FIX_TRAIN_DATA_EACH_EPOCH = False


#### SPECIFIC
from dataset.gp_acquisition_dataset_manager import (
    GET_TRAIN_TRUE_GP_STATS,
    GET_TEST_TRUE_GP_STATS
)


class AcquisitionFunctionSingleTrainer(SingleTrainer):
    train_or_test_loop_class = AcquisitionFunctionTrainOrTestLoop

    @classmethod
    def get_test_dataloader(cls, test_ds, batch_size):
        return test_ds.get_dataloader(batch_size=batch_size, drop_last=False)

    @classmethod
    def get_train_dataloaders(
        cls,
        get_train_stats_after_training,
        train_dataset,
        batch_size,
        test_during_training,
        small_test_dataset
    ):
        # If get_train_stats_after_training=True, then we are running through the
        # train dataset twice each epoch. If furthermore the training data is not
        # fixed, then with these two runs through it, the data will be different.
        # But we'd like to have the stats of during training vs after training
        # directly comparable, so we will freeze the data with each epoch.
        need_fix_train_data = get_train_stats_after_training and not train_dataset.data_is_fixed
        fix_train_dataset_each_epoch = need_fix_train_data and _FIX_TRAIN_DATA_EACH_EPOCH

        if fix_train_dataset_each_epoch:
            train_dataloader = None
            def per_epoch_get_train_dataloader():
                return train_dataset.fix_samples(lazy=True) \
                    .get_dataloader(
                        batch_size=batch_size, drop_last=False, cache_pads=False)
        else:
            train_dataloader = train_dataset.get_dataloader(
                batch_size=batch_size, drop_last=False)
            per_epoch_get_train_dataloader = None
        
        if need_fix_train_data and not _FIX_TRAIN_DATA_EACH_EPOCH:
            if test_during_training:
                num = len(small_test_dataset)
            else:
                num = None
            train_dataset_eval_dataloader = train_dataset \
                .fix_samples(n_realizations=num, lazy=False) \
                .get_dataloader(batch_size=batch_size, drop_last=False, cache_pads=True)
        else:
            train_dataset_eval_dataloader = train_dataloader
        
        return (
            train_dataloader, per_epoch_get_train_dataloader, train_dataset_eval_dataloader)

    @classmethod
    def print_stats(cls,
                    stats:dict,
                    dataset_name, method,
                    extra_kwargs:SimpleNamespace,
                    is_test:bool=False):
        n_cand = extra_kwargs.test_n_cand if is_test else extra_kwargs.train_n_cand
        print_dataset_ei = n_cand > 1
        gi_loss_normalization = extra_kwargs.gi_loss_normalization

        print(f'{dataset_name}:')
        if print_dataset_ei:
            print('Expected 1-step improvement:')
            things_to_print = [
                ('ei_softmax', 'NN (softmax)', True),
                ('maxei', 'NN (max)', True),
                ('true_gp_ei_maxei', 'True GP EI', False),
                ('map_gp_ei_maxei', 'MAP GP EI', True),
                ('true_gp_gi_maxei', 'True GP GI', True),
                ('ei_random_search', 'Random search', True),
                ('ei_ideal', 'Ideal', True),
                ('avg_normalized_entropy', 'Avg norm. entropy', False)]
            cls._print_stat_summary(
                stats, things_to_print, best_stat='true_gp_ei_maxei',
                inverse_ratio=False, sqrt_ratio=False)
        
        if method == 'mse_ei':
            print('Improvement MSE:')
            things_to_print = [
                ('mse', 'NN', True),
                ('true_gp_ei_mse', 'True GP EI', False),
                ('map_gp_ei_mse', 'MAP GP EI', True),
                ('mse_always_predict_0', 'Always predict 0', True)]
            cls._print_stat_summary(
                stats, things_to_print, best_stat='true_gp_ei_mse',
                inverse_ratio=True, sqrt_ratio=True, ratio_name='RMSE Ratio')
        elif method == 'gittins':
            print('Gittins index loss:')
            tmp = f'_normalized_{gi_loss_normalization}' if gi_loss_normalization is not None else ''
            things_to_print = [
                ('gittins_loss' + tmp, 'NN', True),
                ('true_gp_gi_gittins_loss' + tmp, 'True GP GI', False)
            ]
            cls._print_stat_summary(
                stats, things_to_print, best_stat='true_gp_gi',
                inverse_ratio=True, sqrt_ratio=False)
    
    @classmethod
    def _print_stat_summary(
            cls, stats, things_to_print, best_stat:Optional[str]=None,
            inverse_ratio=False, sqrt_ratio=False, ratio_name='Ratio'):
        
        # Prepare the rows to print
        best_val = None if best_stat is None else stats.get(best_stat)
        rows = []
        for stat_key, stat_print_name, print_ratio in things_to_print:
            if stat_key in stats:
                val = stats[stat_key]
                this_thing = [stat_print_name+':', f'{val:>8f}']
                if best_val is not None and print_ratio:
                    ratio = best_val / val if inverse_ratio else val / best_val
                    if sqrt_ratio:
                        ratio = math.sqrt(ratio)
                    this_thing.extend([f'  {ratio_name}:', f'{ratio:>8f}'])
                rows.append(this_thing)
        
        # Print the rows
        prefix = "  "
        # Calculate the maximum width for each column
        col_widths = [
            max(len(row[col_idx]) for row in rows if len(row) > col_idx)
            for col_idx in range(max(map(len, rows)))]

        # Print each row with appropriate spacing
        row_strings = [
            prefix + " ".join(
                col_val.rjust(col_widths[col_idx])
                for col_idx, col_val in enumerate(row)
            ) for row in rows]
        print("\n".join(row_strings))

    @classmethod
    def get_stat_name_and_minimize_flag(cls, method, extra_kwargs):
        use_maxei = extra_kwargs.use_maxei
        gi_loss_normalization = extra_kwargs.gi_loss_normalization
        if not isinstance(use_maxei, bool):
            raise ValueError("use_maxei should be a boolean")
        if use_maxei:
            stat_name = "maxei"
            minimize_stat = False
        elif method == "policy_gradient":
            stat_name = "ei_softmax"
            minimize_stat = False
        elif method == "mse_ei":
            stat_name = "mse"
            minimize_stat = True
        elif method == "gittins":
            stat_name = "gittins_loss" + \
                (f"_normalized_{gi_loss_normalization}" \
                if gi_loss_normalization is not None else "")
            minimize_stat = True
        return stat_name, minimize_stat

    @classmethod
    def get_extra_kwargs_namespace(
        cls, args, train_dataset, test_dataset, small_test_dataset) -> SimpleNamespace:
        dataset_type = getattr(args, 'dataset_type', 'gp')
        ret = SimpleNamespace(
            alpha_increment=args.alpha_increment,
            gi_loss_normalization=args.gi_loss_normalization,
            # These both default to reasonable values depending on whether the
            # acquisition datasets are fixed
            get_train_true_gp_stats=GET_TRAIN_TRUE_GP_STATS and dataset_type == 'gp',
            get_test_true_gp_stats=GET_TEST_TRUE_GP_STATS and dataset_type == 'gp',
            get_train_map_gp_stats=False,
            get_test_map_gp_stats=None,
            # evaluation metric
            use_maxei=args.use_maxei
        )
        ## (This is code from before -- I don't think it actually does anything anymore)
        # Due to this, need to explicitly set the default value here because train_or_test_loop
        # won't get it right because we'll fix the data even though it isn't fixed
        if ret.get_train_true_gp_stats is None:
            ret.get_train_true_gp_stats = train_dataset.has_models and train_dataset.data_is_fixed
        
        ret.train_n_cand = next(iter(train_dataset)).x_cand.size(0)
        ret.test_n_cand = next(iter(test_dataset)).x_cand.size(0)

        return ret
    
    @classmethod
    def get_train_stats_while_training_kwargs(cls, extra_kwargs: SimpleNamespace):
        return dict(
            alpha_increment=extra_kwargs.alpha_increment,
            gi_loss_normalization=extra_kwargs.gi_loss_normalization,
            get_true_gp_stats=extra_kwargs.get_train_true_gp_stats,
            get_map_gp_stats=extra_kwargs.get_train_map_gp_stats,
            get_basic_stats=True
        )

    @classmethod
    def get_train_stats_after_training_kwargs(cls, extra_kwargs: SimpleNamespace):
        return dict(
            gi_loss_normalization=extra_kwargs.gi_loss_normalization,
            # Don't need to compute non-NN stats because already computed them
            # while training, and we ensured that the train dataset is fixed for this epoch.
            get_true_gp_stats=False,
            get_map_gp_stats=False,
            get_basic_stats=False
        )

    @classmethod
    def get_test_during_training_kwargs(cls, extra_kwargs: SimpleNamespace):
        return dict(
            gi_loss_normalization=extra_kwargs.gi_loss_normalization,
            get_true_gp_stats=extra_kwargs.get_test_true_gp_stats,
            get_map_gp_stats=extra_kwargs.get_test_map_gp_stats,
            get_basic_stats=True
        )

    @classmethod
    def get_test_after_training_kwargs(
        cls, extra_kwargs: SimpleNamespace, test_dataset):
        # Consider if the test datasets are not fixed.
        # Then the test-after-training default is not good.
        if extra_kwargs.get_test_true_gp_stats is None and \
            not test_dataset.data_is_fixed and test_dataset.has_models:
            get_test_true_gp_stats_after_training = True
        else:
            get_test_true_gp_stats_after_training = extra_kwargs.get_test_true_gp_stats

        return dict(
            gi_loss_normalization=extra_kwargs.gi_loss_normalization,
            get_true_gp_stats=get_test_true_gp_stats_after_training,
            get_map_gp_stats=extra_kwargs.get_test_map_gp_stats,
            get_basic_stats=True
        )
    
    @classmethod
    def get_final_test_stats_kwargs(cls, extra_kwargs: SimpleNamespace):
        return dict(
            get_true_gp_stats=extra_kwargs.get_test_true_gp_stats,
            get_map_gp_stats=False,
            get_basic_stats=True,
            alpha_increment=extra_kwargs.alpha_increment,
            gi_loss_normalization=extra_kwargs.gi_loss_normalization
        )
    
    @classmethod
    def create_train_test_datasets_from_args(cls, args):
        return create_train_test_acquisition_datasets_from_args(args)
    
    @classmethod
    def is_non_nn_stat_name(cls, stat_name: str) -> bool:
        return (stat_name in _BASIC_STATS or
                stat_name.startswith("true_gp") or
                stat_name.startswith("map_gp"))

    @classmethod
    def extra_checks(cls, extra_kwargs, test_during_training, test_after_training):
        if test_during_training or test_after_training:
            if extra_kwargs.get_test_map_gp_stats is None:
                extra_kwargs.get_test_map_gp_stats = False # default
        elif extra_kwargs.get_test_true_gp_stats or extra_kwargs.get_test_map_gp_stats:
            raise ValueError("Can't get GP stats of test dataset because there is none specified")

_single_trainer = AcquisitionFunctionSingleTrainer(ACQF_NET_SAVING)
single_train = _single_trainer.single_train
