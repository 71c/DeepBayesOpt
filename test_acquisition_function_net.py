import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
from function_samples_dataset import GaussianProcessRandomDataset
from acquisition_dataset import FunctionSamplesAcquisitionDataset
from utils import get_uniform_randint_generator, get_loguniform_randint_generator, pad_tensor
from acquisition_function_net import AcquisitionFunctionNetV1, AcquisitionFunctionNetV2, AcquisitionFunctionNetV3


# This means whether n history points is log-uniform
# or whether the total number of points is log-uniform
LOGUNIFORM = True

FIX_N_CANDIDATES = True

if FIX_N_CANDIDATES:
    # Number of candidate points. More relevant for the policy gradient case.
    # Doesn't matter that much for the MSE EI case; for MSE, could just set to 1.
    N_CANDIDATES = 50
    MIN_HISTORY = 1
    MAX_HISTORY_ = 8

else:
    MIN_N_CANDIDATES = 5
    MIN_POINTS = MIN_N_CANDIDATES + 1
    MAX_POINTS = 20

    if LOGUNIFORM:
        n_datapoints_random_gen = get_loguniform_randint_generator(
            MIN_POINTS, MAX_POINTS, pre_offset=3.0, offset=0)
    else:
        n_datapoints_random_gen = get_uniform_randint_generator(
            MIN_POINTS, MAX_POINTS)

BATCH_SIZE = 128
N_BATCHES = 100
EVERY_N_BATCHES = 10
EPOCHS = 10

# dimension of the optimization problem
DIMENSION = 1
# whether to randomize the GP parameters for training data
RANDOMIZE_PARAMS = False
# choose either "uniform" or "normal" (or a custom distribution)
XVALUE_DISTRIBUTION = "uniform"

# True for the softmax thing, False for MSE
POLICY_GRADIENT = True

# Only used if POLICY_GRADIENT is True
INCLUDE_ALPHA = True
# Following 3 are only used if both POLICY_GRADIENT and INCLUDE_ALPHA are True
LEARN_ALPHA = True
INITIAL_ALPHA = 1.0
ALPHA_INCREMENT = None # equivalent to 0.0

# Whether to train to predict the EI rather than predict the I
# Only used if POLICY_GRADIENT is False
TRAIN_WITH_EI = False

# Whether to fit maximum a posteriori GP for testing
FIT_MAP_GP = False

# Whether to train the model. If False, will load a saved model.
TRAIN = False
# Whether to load a saved model to train
LOAD_SAVED_MODEL_TO_TRAIN = False


SOFTMAX = True

def check_model(dataloader, model):
    params_with_differing_grads = set()
    all_outputs_same = True

    for x_hist, y_hist, x_cand, y_cand, hist_mask, cand_mask, models in dataloader:
        output = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, softmax=SOFTMAX)

        dummy_loss = output.sum()
        dummy_loss.backward()
        grads_dict = {name: param.grad for name, param in model.named_parameters()}

        n_hist = x_hist.size(-2)
        n_cand = x_cand.size(-2)

        model.zero_grad()

        outputs = []
        for i in range(BATCH_SIZE):
            x_hist_i_padded = x_hist[i] # shape (n_hist, dimension)
            y_hist_i_padded = y_hist[i] # shape (n_hist,)
            x_cand_i_padded = x_cand[i] # shape (n_cand, dimension)
            hist_mask_i_padded = None if hist_mask is None else hist_mask[i] # shape (n_hist,)
            cand_mask_i_padded = None if cand_mask is None else cand_mask[i] # shape (n_cand,)

            n_hist_i = n_hist if hist_mask is None else hist_mask_i_padded.sum().item()
            n_cand_i = n_cand if cand_mask is None else cand_mask_i_padded.sum().item()

            x_hist_i = x_hist_i_padded[:n_hist_i]
            y_hist_i = y_hist_i_padded[:n_hist_i]
            x_cand_i = x_cand_i_padded[:n_cand_i]
            hist_mask_i = None if hist_mask_i_padded is None else hist_mask_i_padded[:n_hist_i]
            cand_mask_i = None if cand_mask_i_padded is None else cand_mask_i_padded[:n_cand_i]

            output_i = model(x_hist_i, y_hist_i, x_cand_i, hist_mask_i, cand_mask_i, softmax=SOFTMAX)
            output_i_padded = pad_tensor(output_i, n_cand, 0, add_mask=False)
            outputs.append(output_i_padded)
        output_unbatched = torch.stack(outputs)
        
        dummy_loss_unbatched = output_unbatched.sum()
        dummy_loss_unbatched.backward()
        grads_dict_unbatched = {
            name: param.grad for name, param in model.named_parameters()}

        output_same = torch.allclose(output, output_unbatched)
        if not output_same:
            print("Outputs are different.")
            print(output)
            print(output_unbatched)
        all_outputs_same = all_outputs_same and output_same

        for name in grads_dict:
            if not torch.allclose(grads_dict[name], grads_dict_unbatched[name]):
                if name not in params_with_differing_grads:
                    print(f"Gradients for {name} are different.")
                    print(grads_dict[name])
                    print(grads_dict_unbatched[name])
                    params_with_differing_grads.add(name)
        
        model.zero_grad()
    
    return all_outputs_same, params_with_differing_grads


MAX_HISTORY = 30 # roughly the maximum history length
BATCH_SIZE = 5
N_BATCHES = 10

history_length_gen = get_loguniform_randint_generator(
    2, MAX_HISTORY, pre_offset=3.0, offset=0)

def n_datapoints_random_gen(size=1):
    return 2 * history_length_gen(size)

dataset = GaussianProcessRandomDataset(
    dimension=DIMENSION, n_datapoints_random_gen=n_datapoints_random_gen,
    observation_noise=False, set_random_model_train_data=False,
    xvalue_distribution="normal",
    device=device,
    dataset_size=BATCH_SIZE * N_BATCHES)

aq_dataset = FunctionSamplesAcquisitionDataset(
    dataset, n_candidate_points="uniform", n_samples="all",
    give_improvements=False, min_n_candidates=2)

dataloader = aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True)


model = AcquisitionFunctionNetV3(DIMENSION,
                                 history_enc_hidden_dims=[32, 32], pooling="max",
                 encoded_history_dim=32,
                 mean_enc_hidden_dims=[32, 32], mean_dim=8,
                 std_enc_hidden_dims=[32, 32], std_dim=8,
                 aq_func_hidden_dims=[32, 32], layer_norm=False,
                 layer_norm_at_end_mlp=False, include_y=True,
                 include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
                                 learn_alpha=LEARN_ALPHA,
                                 initial_alpha=INITIAL_ALPHA).to(device)

# model = AcquisitionFunctionNetV1(DIMENSION,
#                                  pooling="max",
#                                  history_enc_hidden_dims=[32, 32],
#                                  encoded_history_dim=32,
#                                  aq_func_hidden_dims=[32, 32],
#                                  include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
#                                  learn_alpha=LEARN_ALPHA,
#                                  initial_alpha=INITIAL_ALPHA,
#                                  layer_norm_pointnet=True,
#                                  layer_norm_before_end_mlp=True,
#                                  layer_norm_at_end_mlp=False).to(device)

# model = AcquisitionFunctionNetV1(DIMENSION)
print(model)

script_dir = os.path.dirname(os.path.abspath(__file__))
# training_info = f"batchsize{BATCH_SIZE}_batches_per_epoch{N_BATCHES}_epochs{EPOCHS}"
model_class_name = model.__class__.__name__
loss_str = 'policy_gradient_myopic' if POLICY_GRADIENT else 'ei'
file_name = f"acquisition_function_net_{model_class_name}_{DIMENSION}d_{loss_str}_{'random' if RANDOMIZE_PARAMS else 'fixed'}_kernel_{XVALUE_DISTRIBUTION}_x"
if FIX_N_CANDIDATES:
    file_name += f"_history{MIN_HISTORY}-{MAX_HISTORY_}_{'loguniform' if LOGUNIFORM else 'uniform'}_{N_CANDIDATES}cand.pth"
else:
    file_name += f"_points{MIN_POINTS}-{MAX_POINTS}_{'loguniform' if LOGUNIFORM else 'uniform'}.pth"

print(f"Model file: {file_name}")
model_path = os.path.join(script_dir, file_name)

model.load_state_dict(torch.load(model_path))


all_outputs_same, params_with_differing_grads = check_model(dataloader, model)

if all_outputs_same:
    print("All outputs are the same.")
else:
    print("Not all outputs are the same.")

if len(params_with_differing_grads) == 0:
    print("All gradients are the same.")
else:
    print("Not all gradients are the same. Those that differ are:")
    print(params_with_differing_grads)

