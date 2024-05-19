import torch
from torch.distributions import Uniform, Normal, Independent, Distribution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from generate_gp_data import GaussianProcessRandomDataset, TrainAcquisitionFunctionDataset
from utils import get_uniform_randint_generator, loguniform_randint, get_loguniform_randint_generator
from acquisition_function_net import AcquisitionFunctionNet


MAX_HISTORY = 30 # roughly the maximum history length
BATCH_SIZE = 2
N_BATCHES = 10
DIMENSION = 6

history_length_gen = get_loguniform_randint_generator(
    2, MAX_HISTORY, pre_offset=3.0, offset=0)

def n_datapoints_random_gen(size=1):
    return 2 * history_length_gen(size)


m = Normal(torch.zeros(DIMENSION, device=device),
           torch.ones(DIMENSION, device=device))
xvalue_distribution = Independent(m, 1)

dataset = GaussianProcessRandomDataset(
    dimension=DIMENSION, n_datapoints_random_gen=n_datapoints_random_gen,
    observation_noise=False, set_random_model_train_data=False,
    xvalue_distribution=xvalue_distribution,
    device=device,
    dataset_size=BATCH_SIZE * N_BATCHES)

aq_dataset = TrainAcquisitionFunctionDataset(
    dataset, n_candidate_points="binomial", n_samples="all",
    give_improvements=False, min_n_candidates=2)

dataloader = aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True)


model = AcquisitionFunctionNet(DIMENSION).to(device)
print(model)


for x_hist, y_hist, x_cand, y_cand, hist_mask, cand_mask, models in dataloader:
    output = model(x_hist, y_hist, x_cand, hist_mask, cand_mask)
    # print(y_cand.shape, output.shape)

    # print(x_hist.shape, x_cand.shape)
    print(output.shape)
    print(output)

    print(torch.mean(output, dim=0))
    print(torch.std(output, dim=0))
    print(torch.mean(output, dim=1))
    print(torch.std(output, dim=1))
    print()
