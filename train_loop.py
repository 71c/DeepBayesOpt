from utils import to_device, unsupported_improvements
from train_acquisition_function_net import myopic_policy_gradient_ei, mse_loss, get_average_normalized_entropy
from acquisition_dataset import AcquisitionDataset


def train_loop(dataloader, model, optimizer, every_n_batches=10,
               policy_gradient=False, alpha_increment=None):
    """Trains the acquisition function network.

    Args:
        dataloader (torch.utils.data.DataLoader):
            The dataloader containing the training data.
        model (AcquisitionFunctionNet):
            The acquisition function network model.
        optimizer (torch.optim.Optimizer):
            The optimizer used for training.
        every_n_batches (int, optional, default=10):
            The frequency at which to print the training progress.
        policy_gradient (bool, default=False):
            Whether to use policy gradient loss
        alpha_increment (float, default=None):
            The amount to increase alpha by after each batch.
        train_with_ei:
            Whether to train to predict the EI rather than predict the I.
            Only used if policy_gradient is False.
    """
    if not isinstance(dataloader.dataset, AcquisitionDataset):
        raise ValueError("The dataloader must contain an AcquisitionDataset")
    
    model.train()

    n_batches = len(dataloader)
    # average_train_loss = 0.

    for i, batch in enumerate(unsupported_improvements(dataloader)):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask = batch.tuple_no_model

        # if i % every_n_batches == 0:
        #     print("Test")
        # continue

        # print(type(models))
        # for gp_model in models:
        #     print(gp_model)
        #     for name, param in gp_model.named_parameters():
        #         print(name, param)
        #     print()
        #     print("lengthscale:", gp_model.covar_module.base_kernel.lengthscale)
        #     print("outputscale:", gp_model.covar_module.outputscale)
        #     print("mean:", gp_model.mean_module.constant)
        # print()
        # exit()

        if policy_gradient:
            output = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=False, softmax=True)
            loss = -myopic_policy_gradient_ei(output, improvements)
        else:
            output = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=True, softmax=False)
            loss = mse_loss(output, improvements, cand_mask)
        
        # average_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model.includes_alpha and alpha_increment is not None:
            model.set_alpha(model.get_alpha() + alpha_increment)

        if i % every_n_batches == 0:
            prefix = "Expected 1-step improvement" if policy_gradient else "MSE"
            
            suffix = ""
            if policy_gradient:
                avg_normalized_entropy = get_average_normalized_entropy(output, mask=cand_mask).item()
                suffix += f", avg normalized entropy={avg_normalized_entropy:>7f}"
            if model.includes_alpha:
                suffix += f", alpha={model.get_alpha():>7f}"
            
            loss_value = -loss.item() if policy_gradient else loss.item()
            print(f"{prefix}: {loss_value:>7f}{suffix}  [{i+1:>4d}/{n_batches:>4d}]")

    # multiplier = -1 if policy_gradient else 1
    # average_train_loss /= multiplier * n_batches

    # print(f"Average train loss: {average_train_loss:>7f}")
    # return average_train_loss
