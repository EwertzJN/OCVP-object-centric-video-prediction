"""
Hungarian algorithm and other methods for data association and set matching
"""

from scipy.optimize import linear_sum_assignment as linear_assignment
import torch
import torchmetrics

from ..CONFIG import CONFIG

MAX = CONFIG["epsilon_max"]
EPS = CONFIG["epsilon_min"]


@torch.no_grad()
def batch_pairwise_matching(dist_tensor, maximize, method="hungarian"):
    """
    Computing the pairwise matches for all elements in a mini-batch
    
    Args:
    -----
    dist_tensor: torch tensor
        Tensor with the distance/scores used for matching. Shape is (B, N_inputs, N_targets).
    maximize: torch tensor
        Whether the matching algorithm must perform the data association based on maximizing
        a score or minimizing a cost.
    method: string
        Type of data association used: either 'hungarian' or 'greedy'

    Returns:
    --------
    all_matches: torch Tensor
        Found mathcing indices. Shape is (B, ...)
    """
    assert method in ["hungarian", "greedy"], f"{method = } must be either 'hungarian' or 'greedy'..."
    all_matches = []
    for dist_matrix in dist_tensor:
        cur_matches = pairwise_matching(dist_matrix.cpu(), method=method, maximize=maximize)
        all_matches.append(cur_matches)
    all_matches = torch.stack(all_matches).float()
    return all_matches


@torch.no_grad()
def pairwise_matching(dist_matrix, method="hungarian", maximize=False):
    """
    Computing the pairwise matches between slots

    Args:
    -----
    dist_matrix: torch.Tensor (N_Slots, N_Slots)
        computing the pairwise matches between slots from img1 and img2
    """
    assert method in ["hungarian", "greedy"], f"Unrecognized matching method: {method}"
    dist_matrix[dist_matrix > MAX] = MAX
    if(method == "hungarian"):
        matched_idx = linear_assignment(dist_matrix, maximize=maximize)
        matched_idx = torch.stack([torch.from_numpy(matched_idx[0]), torch.from_numpy(matched_idx[1])]).T
    elif(method == "greedy"):
        matched_idx = greedy_assignment(dist_matrix, maximize)
    return matched_idx.float()


def greedy_assignment(dist_matrix, maximize):
    """
    Computing pairwise assignments between slots in a rowwise greedy manner
    """
    func = torch.argmax if maximize else torch.argmin
    
    matched_indices = []
    for i in range(dist_matrix.shape[0]):
        j = func(dist_matrix[i]).item()
        dist_matrix[:, j] = EPS
        matched_indices.append(torch.Tensor([i, j]))
    matched_indices = torch.stack(matched_indices)
    return matched_indices


def reorder_given_ids(tokens, ids):
    """
    Reordering a set of tensors given the position ids.
    
    Args:
    -----
    tokens: torch tensor
        Tokens to re-order. Shape is (B, N, D).
    ids: torch tensor
        Tensor with the ids indicating which tensor goes where.
        Shape is (B, N, 2)  --> VERIFY THIS!
    """
    
    # TODO
    
    return tokens


def batch_pairwise_mse(preds1, preds2):
    """
    Computing the pairwise MSE between two sets of tensors.
    
    Args:
    -----
    preds1, preds2: torch tensors
        Tensors to compute the distances of. Shape is (B, N, D)
        
    Returns:
    --------
    pairwise_mse: torch tensor
        Pairwise distances. Shape is (B, N, N)
    """
    pairwise_mse = torch.sum((preds1.unsqueeze(2) - preds2.unsqueeze(1)) ** 2, dim=-1)
    return pairwise_mse


def batch_categorical_matching(preds, targets):
    """
    Computing the pairwise accuracy between one-hot encoded predictions
    and numerical targets
    
    Args:
    -----
    preds: torch tensor
        Predicted categorical values in one-hot encoding. Shape is (B, num_pred_values, num_classes)
    preds: torch tensor
        Target categorical values in integer format. Shape is (B, num_target_values)
        
    Returns:
    --------
    pairwise_costs: torch tensor
        Pairwise cost (0 if equal, 1 otherwise) between values.
        Shape is (B, num_pred_values,)
    """
    # diff = preds.shape[1] - targets.shape[1]
    # if diff != 0:
    #     targets = torch.cat([targets, -1 * torch.ones_like(targets[:, -1:]).repeat(1, diff)], dim=-1)
    pred_ids = torch.argmax(preds, dim=-1)
    pairwise_costs = pred_ids.unsqueeze(2) - targets.unsqueeze(1)
    pairwise_costs[pairwise_costs != 0] = 1
    return pairwise_costs


# FROM HERE ON NOT TESTED YET


def batch_cosine_similarity(slots1, slots2):
    """
    Computing the cosine similarty between two sets of slots in batch mode
    """
    all_scores = []
    for i in range(slots1.shape[0]):
        slot1, slot2 = slots1[i], slots2[i]
        scores = cosine_similarity(slot1, slot2)
        all_scores.append(scores)
    all_scores = torch.stack(all_scores).float()
    return all_scores


def cosine_similarity(slots1, slots2):
    """
    Computing pairwise cosine similariy scores of slots of two images
    """
    return torchmetrics.functional.pairwise_cosine_similarity(slots1, slots2)


@torch.no_grad()
def align_sequence(slot_sequence):
    """
    Aligning a sequence of slots by iterating over consecutie time steps

    Args:
    -----
    slot_sequence: torch tensor
        Batch of slots corresponding to a sequence. Shape is (B, num_frames, num_slots, slot_dim)

    Returns:
    --------
    aligned_slot_seq: torch Tensor
        Input slots sequence, but slots are now temporally aligned.
        For instance, if slot_1 is a red ball at t=0, it should still be the red ball at all time steps.
        Shape is (B, num_frames, num_slots, slot_dim)
    all_sim_scores: torch Tensor
        Pairwise similarity scores between slots from consecutive frames.
        Shape is (B, num_frames-1, num_slots, slot_dim)
    """
    (B, L, num_slots, slot_dim), device = slot_sequence.shape, slot_sequence.device
    aligned_slots = torch.zeros(B, L, num_slots, slot_dim).to(device)
    aligned_slots[:, 0, :, :] = slot_sequence[:, 0, :, :]
    all_sim_scores = []
    for t in range(L-1):
        slots1 = aligned_slots[:, t, :, :].cpu()
        slots2 = slot_sequence[:, t+1, :, :].cpu()
        all_ys, sim_scores = align_slots(slots1, slots2)
        all_sim_scores.append(sim_scores)
        for b, ys in enumerate(all_ys):
            aligned_slots[b, t+1] = slot_sequence[b, t+1][ys]
    all_sim_scores = torch.stack(all_sim_scores, dim=1)
    return aligned_slots, all_sim_scores


@torch.no_grad()
def align_slots(slots1, slots2):
    """
    Aligning two sets of slots given their similarity

    Args:
    -----
    slots1, slots2: torch Tensors
        Slots sets from frames t and t+1, respectively. Shapes are (B, num_slots, slot_dim)

    Returns:
    --------
    all_ys: list of lists
        Lists containing the matching indexes used for reorering the set of slots.
        Shape is [[y11, y12, ..., y1S], [y21, y22, ..., y2S], ..., [yN1, yN2, ..., yNS]]
    cosine_similarity_scores: torch Tensor
        Pairwise similarity score between slots
    """
    B = slots1.shape[0]
    cosine_similarity_scores = batch_cosine_similarity(slots1, slots2)
    batch_matching_matrices = batch_pairwise_matching(cosine_similarity_scores)
    all_ys = []
    for b in range(B):
        matching_matrix = batch_matching_matrices[b]
        ys = []
        for x, y in matching_matrix:
            x, y = int(x), int(y)
            ys.append(y)
        all_ys.append(ys)
    return all_ys, cosine_similarity_scores
