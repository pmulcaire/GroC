import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.checkpoint import checkpoint
from colorama import Fore, Back, Style
import gc

class ScreeningSoftmax(nn.Module):
    """
        Differentiable large vocabulary screening model.

        Args:
            k: <int>, number of clusters of the vocabulary elements to be calculated.
            samples: <int> number of random vocabulary elements to be used for clustering.
            thresh: <float> the threshold to be used for cluster membership.
            freq: <int> the frequency with which the membership assignment is made.
            debug: <bool>, print out messages for debugging.

        Example:
            >>> # within the decoder model forward pass
            >>> screening_model = ScreeningSoftmax(k=100, thresh=0.001)
            >>> weight = screening_model(contexts, weight, targets)
            >>> bias = screening_model.filter(bias)
            >>>
            >>> # within training loop
            >>> logits = torch.mm(output,weight.t()) + bias
            >>> raw_loss = criterion(logits, torch.LongTensor(screening_model.targets_reindexed).cuda())
    """
    def __init__(self, k=100, thresh=0.01, freq=-1, samples=None, emsize=300, ninp=300, debug=False, ntoken=10000, cuda=None, fp16=False):
        super(ScreeningSoftmax, self).__init__()
        self.k = k
        self.samples = samples
        self.thresh = thresh
        self.freq = freq
        self.debug = debug
        self.ntoken = ntoken
        if emsize is not None:
            self.cluster = nn.Linear(emsize, k)
            self.cluster_h = nn.Linear(ninp, k)
        else:
            self.cluster = self.cluster_h = nn.Linear(ninp, k)
        self.cluster_members = None
        self.cuda_on = cuda if cuda is not None else torch.cuda.is_available()

        # add support for fp16
        if fp16:
            self.cluster = self.cluster.half()
            self.cluster_h = self.cluster_h.half()

    def cluster_context(self, contexts):
        """Function to be used for clustering context vectors.

        Args:
            contexts: <Tensor>, tensor which contains the context vectors.
        Returns:
            cluster_masks: <Tensor>, tensor in the dimensionality of the contexts
                           containing the cluster assignments.
        """
        cluster_logits = self.cluster_h(contexts.view(-1, contexts.shape[-1]))
        cluster_masks = self.gumbel_softmax(cluster_logits, tau=1, hard=True)
        return cluster_masks

    def cluster_weights(self, weight, targets, full=False):
        """Function to be used for clustering weight vectors.

        Args:
            weight: <Tensor>,  contains the weight vectors.
            targets: <Tensor>, contains the targets of the current contexts.
        Returns:
            cluster_masks: <Tensor>, cluster assignments in the dimensionality of
                                     the weights given as input.
        """
        # release memory before computing the clusters.
        if not full:
            self.free_memory()
        # get a sample of weights if necessary.
        selected_weights = weight
        selected_ids = []
        if self.samples is not None and not full:
            selected_weights, selected_ids = self.sample_weights(weight, targets)

        # initialize members
        if not hasattr(self, "cluster_members") and not full:
            self.cluster_members = torch.zeros(len(selected_ids), self.k).cuda()
        # cluster the weights.
        clustered_weights = self.cluster(selected_weights)

        # sample from the Gumbel softmax distribution.
        cluster_members = self.gumbel_softmax(clustered_weights, tau=1, hard=True)

        # print stats if debug mode is on / might through nan with fp16
        if self.debug:
            print("[*] #clusters: %d #members: %d" % (int(cluster_members.shape[1]), int(cluster_members.sum(dim=0).mean()) ) )
        return cluster_members.t(), selected_ids

    def sample_weights(self, weight, targets):
        """Function to be used for sampling weight vectors.

        Args:
            weight: <Tensor>, contains the weight vectors.
            targets: <Tensor>, contains the targets of the current contexts.
        Returns:
            selected_weights: <Tensor>, contains the selected weights.
        """
        # sample from the full vocabulary for efficiency
        random_ids = np.unique(targets.tolist() + np.random.randint(weight.shape[0], size=self.samples).tolist()).tolist()
        soft_probs = self.gumbel_softmax(self.cluster(weight[random_ids]).t(), tau=1, hard=False)
        self.sampled_targets = random_ids
        selected_weights = weight[random_ids,:]
        return selected_weights, random_ids

    def get_hash(self, index_array):
        """ Get the hash for quick access of the given array. """
        hash_targets = {}
        for j, val in enumerate(index_array):
            hash_targets[val] = j
        return hash_targets

    def gumbel_softmax(self, y_soft, tau=1, hard=False, eps=1e-10, dim=-1, vocab=False):
        """
            Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.
            This is an optimized version that does better memory handling than pytorch version.
        """
        y_soft += -torch.empty_like(y_soft).exponential_().log()  # ~Gumbel(0,1)
        y_soft /= tau  # ~Gumbel(logits,tau)
        y_soft = y_soft.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            return (torch.zeros_like(y_soft).scatter_(dim, index, 1.0) - y_soft).detach() + y_soft
        else:
            # Reparametrization trick.
            return y_soft
        return ret

    def reindex(self, ids, targets=None):
        """ Reindex the ids according to the filtered ids based on the screening model.
        """
        # get hash with val positions for fast access below.
        if targets is None:
            targets = self.sampled_targets
        filtered_hash = self.get_hash(targets)

        # reindex the targets according to the filtered ids.
        reindexed_ids = torch.LongTensor([filtered_hash[idx]  for idx in ids])

        # load on cuda if necessary
        return reindexed_ids.cuda() if self.cuda_on else reindexed_ids


    def forward(self, output, weight, targets=None, current_batch=None):
        """Basic forward function for the screening model.

        Args:
            output: <Tensor>, contains the input context vectors in batches.
            reassign: <bool>, whether to compute the cluster_weights() function or not.
            targets: <Tensor>, contains the targets of the current contexts.
            current_batch: <int>, current batch id to be used for deciding how
                           often to compute the weight clusters according to freq.
        Returns:
            <Tensor>, contains the filtered weights based on the screening model.
        """
        self.current_batch = current_batch if "current_batch" not in dir(self) else self.current_batch
        self.targets_reindexed = None

        # cluster the context first to get the target cluster mask.
        cluster_masks = self.cluster_context(output)

        # cluster the weights and cache the top candidates per cluster if needed.
        cached = not ((self.current_batch+1) % self.freq == 1 or self.cluster_members \
                 is None or self.freq == 1)

        # compute the cluster membership mask
        if not cached:
            self.cluster_members, selected_ids = self.cluster_weights(weight, targets)
        else:
            self.cluster_members = self.cluster_members.detach()
            if self.samples is not None:
                selected_ids = self.sampled_targets
                inter = set(selected_ids) & set(targets.tolist())
                missing = list(set(targets.tolist()) - inter)
                if len(missing) > 0:
                    additional_members, _ = self.cluster_weights(weight[missing], targets, full=True)
                    self.cluster_members = torch.cat([self.cluster_members, additional_members], dim=1)
                    selected_ids = selected_ids + missing
                    self.sampled_targets = selected_ids
        # multiply the cluster masks with the top members per cluster to get the predictions.
        self.predictions = torch.mm(cluster_masks.sum(dim=0).view(1,-1), self.cluster_members[:, :]).squeeze()

        # keep only the members with nonzero values for further processing.
        if self.samples is not None:
            # ensure that the targets are members in the predicted clusters.
            self.predictions[self.reindex(targets.tolist())] = 1.
            nonzero_entries = self.predictions.nonzero(as_tuple=False).squeeze().cpu()
            self.nonzero_targets = np.array(selected_ids)[nonzero_entries]
        else:
            # ensure that the targets are members in the predicted clusters.
            self.predictions[targets] = 1.
            nonzero_entries = self.predictions.nonzero(as_tuple=False).squeeze().cpu()
            self.nonzero_targets = nonzero_entries

        # choose only the nonzero entries
        self.predictions = self.predictions[nonzero_entries]

        # reindex the targets
        self.targets_reindexed = self.reindex(targets.tolist(), targets=self.nonzero_targets.tolist())

        # free up some space
        del(cluster_masks)
        del(nonzero_entries)
        torch.cuda.empty_cache() if self.cuda_on else None
        return self.filter(weight)

    def filter(self, arr):
        """Screen weights or biases based on the screening model.

        Args:
            arr: <Tensor>, contains weights or biases in the dimensionality of
                 vocabulary to be filtered based on the screening model.

        Returns:
            arr: <Tensor>, contains the subset of weights that were selected by
                 the screening model multiplied by the word prediction mask.
        """
        arr = arr[self.nonzero_targets]
        if len(arr.shape) > 1:
            onehots = self.predictions.expand_as(arr.t()).t()
        else:
            onehots = self.predictions
        result = arr * onehots
        return result

    def free_memory(self):
        """ Free caches to release GPU memory."""
        if hasattr(self, "cluster_members"):
            del (self.cluster_members)
        if hasattr(self, "predictions"):
            del (self.predictions)
        if hasattr(self, "targets"):
            del (self.sampled_targets)
        torch.cuda.empty_cache() if self.cuda_on else None


if __name__ == "__main__":
    # initialize arguments
    batch_size = 4
    seq_len = 20
    vocab_size = 1000000
    samples = 50000
    dim = 300
    cuda = False

    def print_mem():
        allocated = round(torch.cuda.memory_allocated()*0.000001)
        reserved = round(torch.cuda.memory_reserved()*0.000001)
        print("Allocated: %dMB / Reserved: %dMB\n" % (allocated, reserved) )

    # initialize main variables
    contexts = torch.rand((batch_size, seq_len, dim))
    weight = torch.rand((vocab_size, dim))
    bias = torch.rand((vocab_size))
    targets =  torch.randint(vocab_size, (batch_size, seq_len)).view(-1)
    if cuda: #load tensors to cuda if necessary
        contexts = contexts.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        targets = targets.cuda()
    print ("-"*40,"\n")
    print ("***** softmax *****")
    print ("Weight shape: ", list(weight.shape))
    print ("Bias shape: ", list(bias.shape))

    # compute time it takes to compute softmax
    criterion = torch.nn.CrossEntropyLoss()
    contexts = contexts.view(-1, dim)
    start = time.time()
    logits = torch.mm(contexts, weight.t()) + bias
    raw_loss = criterion(logits, targets).item()
    end = time.time()
    base_elapsed = end - start
    print ("Loss: ", raw_loss)
    print ("Time: %.5f sec (%s+0.0%s%s)\n" % (base_elapsed, Fore.WHITE, "%", Style.RESET_ALL) )
    del(logits)
    torch.cuda.empty_cache() if cuda else None
    print_mem()
    print ("-"*40,"\n")

    # demonstrate potential speedup for different thresholds
    for num_clusters in [10, 30, 100, 300, 1000, 3000]:

        print ("***** screening softmax (k=%s) *****" % (str(num_clusters)))
        # create the screening model
        screening_model = ScreeningSoftmax(k=num_clusters, samples=samples, cuda=cuda, emsize=dim, ninp=dim, freq=10, debug=True, ntoken=vocab_size)

        if cuda: # load model to cuda if activated
            screening_model = screening_model.cuda()

        # make a forward pass using the screening model
        weight_new = screening_model(contexts, weight, targets, current_batch=1)
        bias_new = screening_model.filter(bias)
        targets_reiindexed = torch.tensor(screening_model.targets_reindexed)

        if cuda: # load model and indexes to cuda if activated
            bias_new = bias_new.cuda()
            targets_reiindexed = targets_reiindexed.cuda()

        print ("Weight shape: ", list(weight_new.shape))
        print ("Bias shape: ", list(bias_new.shape))

        # compute time it takes to compute softmax after screening
        start = time.time()
        logits = torch.mm(contexts, weight_new.t()) + bias_new
        raw_loss = criterion(logits, targets_reiindexed).item()
        end = time.time()
        elapsed = end - start

        if base_elapsed < elapsed:
            color, sign = Fore.RED, "+"
        else:
            color, sign = Fore.GREEN, ""
        imp = 100 - base_elapsed*100/elapsed
        print ("Loss: ", raw_loss)
        print ("Time: %.5f sec (%s%s%2.1f%s%s)" % (elapsed, color, sign, imp, '%', Style.RESET_ALL) )
        print ("")

        del(weight_new)
        del(bias_new)
        del(logits)
        del(screening_model)
        del(targets_reiindexed)
        del(raw_loss)
        torch.cuda.empty_cache() if cuda else None
        print_mem()

    print ("-"*40,"\n")
