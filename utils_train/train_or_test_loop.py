from typing import Optional
from types import SimpleNamespace
from dataset.acquisition_dataset import AcquisitionDataset
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from botorch.exceptions import UnsupportedError

from utils_general.training.train_or_test_loop import TrainOrTestLoop
from utils_general.training.train_utils import calculate_gittins_loss, GI_NORMALIZATIONS
from utils.utils import calculate_batch_improvement
from utils.exact_gp_computations import (
    calculate_EI_GP_padded_batch, calculate_gi_gp_padded_batch)
from utils_train.acquisition_function_net import (
    AcquisitionFunctionNet, ExpectedImprovementAcquisitionFunctionNet,
    GittinsAcquisitionFunctionNet)
from utils_train.acquisition_function_net_constants import METHODS
from utils_train.train_utils import compute_maxei, get_average_normalized_entropy, mse_loss, myopic_policy_gradient_ei


class AcquisitionFunctionTrainOrTestLoop(TrainOrTestLoop):
    @classmethod
    def train_or_test_loop(
        cls,
        dataloader: DataLoader,
        nn_model: Optional[AcquisitionFunctionNet]=None,
        train:Optional[bool]=None,
        nn_device=None,
        method:Optional[str]=None, # ONLY used when training NN
        verbose:bool=True,
        desc:Optional[str]=None,
        
        n_train_printouts:Optional[int]=None,
        optimizer:Optional[torch.optim.Optimizer]=None,

        # Whether to return None if there is nothing to compute
        return_none=False,

        get_true_gp_stats:Optional[bool]=None,
        get_map_gp_stats:bool=False,
        get_basic_stats:bool=True,
        # Only used when method="mse_ei" or "policy_gradient"
        # (only does anything if method="policy_gradient")
        alpha_increment:Optional[float]=None,
        # Only used when method="gittins" and train=True
        gi_loss_normalization:Optional[str]=None
    ) -> Optional[dict]:
        return super().train_or_test_loop(
            dataloader=dataloader,
            nn_model=nn_model,
            train=train,
            nn_device=nn_device,
            method=method,
            verbose=verbose,
            desc=desc,
            n_train_printouts=n_train_printouts,
            optimizer=optimizer,
            return_none=return_none,

            get_true_gp_stats=get_true_gp_stats,
            get_map_gp_stats=get_map_gp_stats,
            get_basic_stats=get_basic_stats,
            alpha_increment=alpha_increment,
            gi_loss_normalization=gi_loss_normalization
        )

    methods = METHODS
    module_class = AcquisitionFunctionNet
    
    def __init__(
            self,
            dataloader: DataLoader,
            nn_model: Optional[AcquisitionFunctionNet]=None,
            train:Optional[bool]=None,
            nn_device=None,
            method:Optional[str]=None, # ONLY used when training NN
            verbose:bool=True,
            desc:Optional[str]=None,
            
            n_train_printouts:Optional[int]=None,
            optimizer:Optional[torch.optim.Optimizer]=None,

            **specific_kwargs):
        super().__init__(
            dataloader=dataloader,
            nn_model=nn_model,
            train=train,
            nn_device=nn_device,
            method=method,
            verbose=verbose,
            desc=desc,
            n_train_printouts=n_train_printouts,
            optimizer=optimizer,
            **specific_kwargs
        )
        #### Extract from specific_kwargs
        self.get_true_gp_stats = specific_kwargs.get("get_true_gp_stats")
        self.get_map_gp_stats = specific_kwargs.get("get_map_gp_stats", False)
        self.get_basic_stats = specific_kwargs.get("get_basic_stats", True)
        self.alpha_increment = specific_kwargs.get("alpha_increment")
        self.gi_loss_normalization = specific_kwargs.get("gi_loss_normalization")
        
        ### Validate inputs
        dataset = self.dataset
        if not isinstance(dataset, AcquisitionDataset):
            raise ValueError("The dataloader must contain an AcquisitionDataset")
        if nn_model is not None: # evaluating a NN model
            if method == "gittins":
                if not isinstance(nn_model, GittinsAcquisitionFunctionNet):
                    raise ValueError("nn_model must be a GittinsAcquisitionFunctionNet "
                                    "instance if method='gittins'")
                if nn_model.costs_in_history:
                    raise UnsupportedError("nn_model.costs_in_history=True is currently not"
                                        " supported for method='gittins'")
                if nn_model.cost_is_input:
                    raise UnsupportedError("nn_model.cost_is_input=True is currently not"
                                        " supported for method='gittins'")
                self.nnei = False
            elif method == 'policy_gradient' or method == 'mse_ei':
                if not isinstance(nn_model, ExpectedImprovementAcquisitionFunctionNet):
                    raise ValueError(
                        "nn_model must be a ExpectedImprovementAcquisitionFunctionNet "
                        "instance if method='policy_gradient' or method='mse_ei'")
                self.nnei = True
            else:
                raise UnsupportedError(f"method '{method}' is not supported")
        self.has_models = dataset.has_models
        if self.get_true_gp_stats is None:
            self.get_true_gp_stats = self.has_models and dataset.data_is_fixed
        if not isinstance(self.get_true_gp_stats, bool):
            raise ValueError("get_true_gp_stats must be a boolean")
        if not isinstance(self.get_map_gp_stats, bool):
            raise ValueError("get_map_gp_stats must be a boolean")
        if not isinstance(self.get_basic_stats, bool):
            raise ValueError("get_basic_stats must be a boolean")
        if not self.has_models:
            if self.get_true_gp_stats:
                raise ValueError("get_true_gp_stats must be False if no models are present")
            if self.get_map_gp_stats:
                raise ValueError("get_map_gp_stats must be False if no models are present")
        if nn_model is not None and train and self.alpha_increment is not None:
            if not (isinstance(self.alpha_increment, float) and self.alpha_increment >= 0):
                raise ValueError("alpha_increment must be a positive float")
        if not train and self.alpha_increment is not None:
            raise ValueError("alpha_increment must not be specified if train != True")
        
        self.has_true_gp_stats = hasattr(dataset, "_cached_true_gp_stats")
        self.has_map_gp_stats = hasattr(dataset, "_cached_map_gp_stats")
        self.has_basic_stats = hasattr(dataset, "_cached_basic_stats")
        if not dataset.data_is_fixed:
            assert not (self.has_true_gp_stats or self.has_map_gp_stats or self.has_basic_stats)
        self.compute_true_gp_stats = self.get_true_gp_stats and not self.has_true_gp_stats
        self.compute_map_gp_stats = self.get_map_gp_stats and not self.has_map_gp_stats
        self.compute_basic_stats = self.get_basic_stats and not self.has_basic_stats

        if self.compute_true_gp_stats:
            self.true_gp_stats_list = []
        if self.compute_map_gp_stats:
            self.map_gp_stats_list = []
        if self.compute_basic_stats:
            self.basic_stats_list = []
        
        self.do_nothing = (self.do_nothing
                           and not self.compute_true_gp_stats
                           and not self.compute_map_gp_stats
                           and not self.compute_basic_stats)

    def get_data_from_batch(self, batch) -> SimpleNamespace:
        x_hist, y_hist, x_cand, vals_cand, hist_mask, cand_mask = batch.tuple_no_model

        n_out_cand = vals_cand.size(-1)
        if not (n_out_cand == 1 or n_out_cand == 2):
            raise ValueError("Expected either 1 or 2 output values per candidate, "
                            f"but got {n_out_cand} output values")

        vals_cand_0 = vals_cand if n_out_cand == 1 else vals_cand[..., 0].unsqueeze(-1)
        
        ret = SimpleNamespace(
            x_hist=x_hist,
            y_hist=y_hist,
            x_cand=x_cand,
            vals_cand=vals_cand,
            hist_mask=hist_mask,
            cand_mask=cand_mask,
            n_out_cand=n_out_cand,
            vals_cand_0=vals_cand_0,
        )
        if batch.give_improvements:
            ret.improvements = vals_cand_0
        else:
            ret.y_cand = vals_cand_0
            ret.improvements = calculate_batch_improvement(y_hist, ret.y_cand, hist_mask, cand_mask)
        ret.batch_size = ret.improvements.size(0)
        if batch.has_model:
            ret.models = batch.model
        return ret

    def get_data_from_batch_for_nn(self, batch) -> SimpleNamespace:
        (x_hist, y_hist, x_cand, vals_cand,
        hist_mask, cand_mask) = batch.to(self.nn_device).tuple_no_model

        method = self.method
        n_out_cand = vals_cand.size(-1)

        # Only check this when we are training the NN
        if method == 'gittins':
            if n_out_cand != 2:
                raise ValueError(f"Gittins index method requires 2 cand-vals (y, lambda), but got {n_out_cand} cand-vals")
        else:
            if n_out_cand != 1:
                raise UnsupportedError(
                    "Expected 1 output value per candidate for training when method is not 'gittins'")
        
        ret = SimpleNamespace(
            x_hist=x_hist,
            y_hist=y_hist,
            x_cand=x_cand,
            hist_mask=hist_mask,
            cand_mask=cand_mask,
        )
        
        if batch.give_improvements:
            if method == 'gittins':
                raise RuntimeError(
                    "Has batch.give_improvements==True but we need the y values for Gittins index loss")
            ret.y_cand = None
            ret.improvements = vals_cand
        else:
            # y_cand shape: batch x n_cand x 1
            if method == 'gittins':
                ret.y_cand = vals_cand[..., 0].unsqueeze(-1)
            else:
                ret.y_cand = vals_cand
            ret.improvements = calculate_batch_improvement(
                y_hist, ret.y_cand, hist_mask, cand_mask)
        ret.batch_size = ret.improvements.size(0)

        if method == 'gittins':
            ret.log_lambdas = vals_cand[..., 1].unsqueeze(-1)
            ret.lambdas = torch.exp(ret.log_lambdas)
        else:
            ret.lambdas = None
        
        return ret

    def evaluate_nn_on_batch(self, batch_data: SimpleNamespace) -> Tensor:
        method = self.method
        nn_model = self.nn_model
        if method == 'gittins':
            if nn_model.variable_lambda:
                lambda_cand = batch_data.log_lambdas
            else:
                lambda_cand = None
            return nn_model(
                batch_data.x_hist, batch_data.y_hist, batch_data.x_cand,
                lambda_cand=lambda_cand,
                hist_mask=batch_data.hist_mask, cand_mask=batch_data.cand_mask,
                is_log=True
            )
        else: # method = 'mse_ei' or 'policy_gradient' (nnei=True)
            return nn_model(
                batch_data.x_hist, batch_data.y_hist, batch_data.x_cand,
                batch_data.hist_mask, batch_data.cand_mask,
                exponentiate=(method == "mse_ei"),
                softmax=(method == "policy_gradient"))

    @classmethod
    def _compute_acquisition_output_batch_stats(
            cls, output, cand_mask, method:str,
            improvements=None,
            y_cand=None, lambdas=None, normalize=None,
            return_loss:bool=False, name:str="", reduction="mean"):
        if not isinstance(return_loss, bool):
            raise ValueError("return_loss should be a boolean")
        if not isinstance(name, str):
            raise ValueError("name should be a string")
        if name != "":
            name = name + "_"

        output_detached = output.detach()

        ret = {}
        if method == "policy_gradient":
            if improvements is None:
                raise ValueError(
                    "improvements must be specified for method='policy_gradient'")
            ret[name+"avg_normalized_entropy"] = get_average_normalized_entropy(
                output_detached, mask=cand_mask, reduction=reduction).item()
            ei_softmax = myopic_policy_gradient_ei(
                output if return_loss else output_detached, improvements, reduction)
            ret[name+"ei_softmax"] = ei_softmax.item()
            if return_loss:
                ret[name+"loss"] = -ei_softmax  # Note the negative sign
        elif method == "mse_ei":
            if improvements is None:
                raise ValueError("improvements must be specified for method='mse_ei'")
            mse = mse_loss(output if return_loss else output_detached,
                        improvements, cand_mask, reduction)
            ret[name+"mse"] = mse.item()
            if return_loss:
                ret[name+"loss"] = mse
        elif method == "gittins":
            if y_cand is None:
                raise ValueError("y_cand must be specified for method='gittins'")
            if lambdas is None:
                raise ValueError("lambdas must be specified for method='gittins'")
            normalizes = [None] + GI_NORMALIZATIONS if normalize is None else [normalize]
            for nrmlz in normalizes:
                gittins_loss = calculate_gittins_loss(
                    output if return_loss else output_detached, y_cand, lambdas,
                    costs=None, normalize=nrmlz, mask=cand_mask, reduction=reduction)
                nam = name + "gittins_loss" + (f"_normalized_{nrmlz}" if nrmlz else "")
                ret[nam] = gittins_loss.item()
                # Only set loss for the target normalization (unnormalized when normalize is None)
                if return_loss and nrmlz == normalize:
                    ret["loss"] = gittins_loss

        if improvements is not None:
            ret[name+"maxei"] = compute_maxei(output_detached, improvements,
                                            cand_mask, reduction)

        return ret

    def compute_nn_batch_stats(
            self, nn_output, batch_data: SimpleNamespace) -> dict:
        return self._compute_acquisition_output_batch_stats(
            nn_output, batch_data.cand_mask, self.method,
            improvements=batch_data.improvements,
            y_cand=batch_data.y_cand, lambdas=batch_data.lambdas,
            normalize=self.gi_loss_normalization,
            return_loss=self.train, reduction="sum")

    def perform_post_optimizer_step_updates(self):
        nn_model = self.nn_model
        alpha_increment = self.alpha_increment
        if self.nnei and nn_model.includes_alpha and alpha_increment is not None:
            nn_model.set_alpha(nn_model.get_alpha() + alpha_increment)
    
    def print_train_batch_stats(self, nn_batch_stats: dict, batch_size: int):
        self._print_train_batch_stats(
            nn_batch_stats, self.nn_model, self.method,
            self.batch_index, self.n_training_batches,
            reduction="sum", batch_size=batch_size,
            gi_loss_normalization=self.gi_loss_normalization)

    @classmethod
    def _print_train_batch_stats(cls, nn_batch_stats, nn_model, method,
                                batch_index, n_batches,
                                reduction="mean", batch_size=None,
                                gi_loss_normalization=None):
        if reduction == "mean":
            assert batch_size is None
        elif reduction == "sum":
            assert batch_size is not None
            # convert sum reduction to mean reduction
            nn_batch_stats = {k: v / batch_size for k, v in nn_batch_stats.items()}
        else:
            raise ValueError("'reduction' must be either 'mean' or 'sum'")

        suffix = ""
        if method == 'policy_gradient':
            prefix = "Expected 1-step improvement"
            avg_normalized_entropy = nn_batch_stats["avg_normalized_entropy"]
            suffix += f", avg normalized entropy={avg_normalized_entropy:>7f}"
            loss_value = nn_batch_stats["ei_softmax"]
        elif method == 'mse_ei':
            prefix = "MSE"
            loss_value = nn_batch_stats["mse"]
        elif method == 'gittins':
            prefix = "Gittins index loss"
            loss_value = nn_batch_stats[
                "gittins_loss" + (
                    f'_normalized_{gi_loss_normalization}' \
                        if gi_loss_normalization is not None else '')]
        else:
            raise UnsupportedError(f"method '{method}' is not supported")
        if isinstance(nn_model, ExpectedImprovementAcquisitionFunctionNet):
            if nn_model.includes_alpha:
                suffix += f", alpha={nn_model.get_alpha():>7f}"
            if method == 'mse_ei':
                beta = nn_model.get_beta()
                tau = 1 / beta
                suffix += f", tau={tau:>7f}"
                if nn_model.transform.softplus_batchnorm:
                    const = nn_model.transform.batchnorm.weight.get_value().item()
                    suffix += f", batchnorm constant={const:>7f}"

        print(f"{prefix}: {loss_value:>7f}{suffix}  [{batch_index+1:>4d}/{n_batches:>4d}]")

    def compute_additional_batch_stats(self, batch_data: SimpleNamespace):
        x_hist = batch_data.x_hist
        y_hist = batch_data.y_hist
        x_cand = batch_data.x_cand
        vals_cand = batch_data.vals_cand
        hist_mask = batch_data.hist_mask
        cand_mask = batch_data.cand_mask
        n_out_cand = batch_data.n_out_cand
        vals_cand_0 = batch_data.vals_cand_0
        improvements = batch_data.improvements
        with torch.no_grad():
            if self.compute_true_gp_stats:
                # Calculate true GP EI stats
                ei_values_true_model = calculate_EI_GP_padded_batch(
                    x_hist, y_hist, x_cand, hist_mask, cand_mask, batch_data.models)

                true_gp_batch_stats = self._compute_acquisition_output_batch_stats(
                    ei_values_true_model, cand_mask, method='mse_ei',
                    improvements=improvements,
                    return_loss=False, name="true_gp_ei", reduction="sum")

                if n_out_cand == 2: # Gittins index
                    log_lambdas = vals_cand[..., 1].unsqueeze(-1)
                    lambdas = torch.exp(log_lambdas)
                    gi_values_true_model = calculate_gi_gp_padded_batch(
                        batch_data.models,
                        x_hist, y_hist, x_cand,
                        lambda_cand=lambdas,
                        hist_mask=hist_mask, cand_mask=cand_mask,
                        is_log=False
                    )
                    # normalize=None here means normalize all options
                    true_gp_batch_stats_gi = self._compute_acquisition_output_batch_stats(
                        gi_values_true_model, cand_mask, method='gittins',
                        improvements=improvements,
                        y_cand=batch_data.y_cand, lambdas=lambdas, normalize=None,
                        return_loss=False, name="true_gp_gi", reduction="sum")
                    true_gp_batch_stats = {**true_gp_batch_stats, **true_gp_batch_stats_gi}

                self.true_gp_stats_list.append(true_gp_batch_stats)
            
            if self.compute_basic_stats:
                # Calculate the E(I) of selecting a point at random,
                # the E(I) of selecting the point with the maximum I (cheating), and
                # the MSE loss of always predicting 0
                if cand_mask is None:
                    random_search_probs = torch.ones_like(vals_cand_0) / vals_cand_0.size(1)
                else:
                    random_search_probs = cand_mask.double() / cand_mask.sum(
                        dim=1, keepdim=True).double()
                self.basic_stats_list.append({
                    "ei_random_search": myopic_policy_gradient_ei(
                        random_search_probs, improvements, reduction="sum").item(),
                    "ei_ideal": compute_maxei(improvements, improvements,
                                            cand_mask, reduction="sum"),
                    "mse_always_predict_0": mse_loss(
                        torch.zeros_like(vals_cand_0), improvements, cand_mask,
                        reduction="sum").item()
                })
        
        if self.compute_map_gp_stats: # I'm not updating this part anymore, I don't care
            # Calculate the MAP GP EI values
            ei_values_map = calculate_EI_GP_padded_batch(
                x_hist, y_hist, x_cand, hist_mask, cand_mask, batch_data.models, fit_params=True)
            map_gp_batch_stats = self._compute_acquisition_output_batch_stats(
                    ei_values_map, cand_mask, method='mse_ei',
                    improvements=improvements, return_loss=False,
                    name="map_gp_ei", reduction="sum")
            self.map_gp_stats_list.append(map_gp_batch_stats)

    def aggregate_stats(self) -> dict:
        ret = super().aggregate_stats()
        
        if self.get_true_gp_stats:
            if not self.has_true_gp_stats:
                true_gp_stats = self.get_average_stats(
                    self.true_gp_stats_list, "sum", self.dataset_length)
                if self.dataset.data_is_fixed:
                    self.dataset._cached_true_gp_stats = true_gp_stats
            else:
                true_gp_stats = self.dataset._cached_true_gp_stats
            ret.update(true_gp_stats)
        if self.get_map_gp_stats:
            if not self.has_map_gp_stats:
                map_gp_stats = self.get_average_stats(
                    self.map_gp_stats_list, "sum", self.dataset_length)
                if self.dataset.data_is_fixed:
                    self.dataset._cached_map_gp_stats = map_gp_stats
            else:
                map_gp_stats = self.dataset._cached_map_gp_stats
            ret.update(map_gp_stats)
        if self.get_basic_stats:
            if not self.has_basic_stats:
                basic_stats = self.get_average_stats(
                    self.basic_stats_list, "sum", self.dataset_length)
                if self.dataset.data_is_fixed:
                    self.dataset._cached_basic_stats = basic_stats
            else:
                basic_stats = self.dataset._cached_basic_stats
            ret.update(basic_stats)
        
        return ret


train_or_test_loop = AcquisitionFunctionTrainOrTestLoop.train_or_test_loop
