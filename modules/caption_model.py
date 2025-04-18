from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.utils as utils


class CaptionModel(nn.Module):
    def __init__(self):
        """Initialize the CaptionModel."""
        super(CaptionModel, self).__init__()

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):
        # Function to compute diversity penalty
        def add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobs = logprobs.clone()
            batch_size = beam_seq_table[0].shape[0]

            if divm > 0:
                change = logprobs.new_zeros(batch_size, logprobs.shape[-1])
                for prev_choice in range(divm):
                    prev_decisions = beam_seq_table[prev_choice][:, :, local_time]  # Nxb
                    for prev_labels in range(bdash):
                        change.scatter_add_(1, prev_decisions[:, prev_labels].unsqueeze(-1),
                                            change.new_ones(batch_size, 1))

                if local_time == 0:
                    logprobs = logprobs - change * diversity_lambda
                else:
                    logprobs = logprobs - self.repeat_tensor(bdash, change) * diversity_lambda

            return logprobs, unaug_logprobs

        # Function to perform one step of beam search
        def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size)  # NxbxV

            if t == 0:
                assert logprobs.shape[1] == 1
                beam_logprobs_sum = beam_logprobs_sum[:, :1]

            # Add current log probs to cumulative log probs
            candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs  # beam_logprobs_sum Nxb logprobs is NxbxV

            # Select top beam_size candidates
            ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
            ys, ix = ys[:, :beam_size], ix[:, :beam_size]
            beam_ix = ix // vocab_size  # Nxb which beam
            selected_ix = ix % vocab_size  # Nxb # which word

            # Calculate state indices
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(
                -1)

            # Update beam sequence and probabilities based on new selections
            if t > 0:
                # Gather according to beam_ix
                assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) ==
                        beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))
                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(
                    beam_seq_logprobs))

            # Append new selected words to beam sequences
            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)  # beam_seq Nxbxl

            # Update cumulative log probabilities
            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
                                logprobs.reshape(batch_size, -1).gather(1, ix)
            assert (beam_logprobs_sum == ys).all()

            # Update log probabilities for current step
            _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
            beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1,
                                                                                      beam_ix.unsqueeze(-1).expand(-1,
                                                                                                                   -1,
                                                                                                                   vocab_size))
            assert (_tmp_beam_logprobs == beam_logprobs).all()

            # Update beam sequence log probabilities
            beam_seq_logprobs = torch.cat([
                beam_seq_logprobs,
                beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)

            # Update states
            new_state = [None for _ in state]
            for _ix in range(len(new_state)):
                #  Copy over state in previous beam q to new beam at vix
                new_state[_ix] = state[_ix][:, state_ix]
            state = new_state

            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state

        # Get beam search parameters from options
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1)  # Should not affect beam search, but will affect diverse beam search
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size  # Beams per group

        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device

        # Initialize beam tables for each group
        beam_seq_table = [torch.LongTensor(batch_size, bdash, 0).to(device) for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(batch_size, bdash, 0, self.vocab_size + 1).to(device) for _ in
                                   range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(batch_size, bdash).to(device) for _ in range(group_size)]

        # Initialize result containers
        done_beams_table = [[[] for __ in range(group_size)] for _ in range(batch_size)]
        state_table = [[_.clone() for _ in init_state] for _ in range(group_size)]
        logprobs_table = [init_logprobs.clone() for _ in range(group_size)]

        # Prepare model arguments for each group
        args = list(args)
        args = utils.split_tensors(group_size, args)  # For each arg, turn (Bbg)x... to (Bb)x(g)x...

        # Special handling for ensemble models
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in
                    range(group_size)]
        else:
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        # Main beam search loop
        for t in range(self.max_seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.max_seq_length + divm - 1:
                    # Get log probabilities for current group
                    logprobs = logprobs_table[divm]

                    # Apply decoding constraints if needed
                    if decoding_constraint and t - divm > 0:
                        logprobs.scatter_(1, beam_seq_table[divm][:, :, t - divm - 1].reshape(-1, 1).to(device),
                                          float('-inf'))

                    # Suppress UNK tokens if requested
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[str(logprobs.size(1) - 1)] == 'UNK':
                        logprobs[:, logprobs.size(1) - 1] = logprobs[:, logprobs.size(1) - 1] - 1000

                    # Add diversity penalty
                    logprobs, unaug_logprobs = add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash)

                    # Perform one step of beam search
                    beam_seq_table[divm], \
                        beam_seq_logprobs_table[divm], \
                        beam_logprobs_sum_table[divm], \
                        state_table[divm] = beam_step(logprobs,
                                                      unaug_logprobs,
                                                      bdash,
                                                      t - divm,
                                                      beam_seq_table[divm],
                                                      beam_seq_logprobs_table[divm],
                                                      beam_logprobs_sum_table[divm],
                                                      state_table[divm])

                    # Check for completed sequences
                    for b in range(batch_size):
                        is_end = beam_seq_table[divm][b, :, t - divm] == self.eos_idx
                        assert beam_seq_table[divm].shape[-1] == t - divm + 1

                        # Force all sequences to end at max_seq_length
                        if t == self.max_seq_length + divm - 1:
                            is_end.fill_(1)

                        # Save completed beams
                        for vix in range(bdash):
                            if is_end[vix]:
                                final_beam = {
                                    'seq': beam_seq_table[divm][b, vix].clone(),
                                    'logps': beam_seq_logprobs_table[divm][b, vix].clone(),
                                    'unaug_p': beam_seq_logprobs_table[divm][b, vix].sum().item(),
                                    'p': beam_logprobs_sum_table[divm][b, vix].item()
                                }
                                # Apply length penalty
                                final_beam['p'] = length_penalty(t - divm + 1, final_beam['p'])
                                done_beams_table[b][divm].append(final_beam)

                        # Penalize completed beams to avoid selecting them again
                        beam_logprobs_sum_table[divm][b, is_end] -= 1000

                    # Get next token and update states
                    it = beam_seq_table[divm][:, :, t - divm].reshape(-1)
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.cuda(), *(
                            args[divm] + [state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)

        # Sort all beams by their log-probabilities
        done_beams_table = [[sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
                            for b in range(batch_size)]
        done_beams = [sum(_, []) for _ in done_beams_table]
        return done_beams

    def sample_next_word(self, logprobs, sample_method, temperature):
        if sample_method == 'greedy':
            # Greedy sampling (argmax)
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()

        elif sample_method == 'gumbel':  # Gumbel softmax sampling
            def sample_gumbel(shape, eps=1e-20):
                """Sample from Gumbel distribution."""
                U = torch.rand(shape).cuda()
                return -torch.log(-torch.log(U + eps) + eps)

            def gumbel_softmax_sample(logits, temperature):
                """Sample from Gumbel-Softmax distribution."""
                y = logits + sample_gumbel(logits.size())
                return F.log_softmax(y / temperature, dim=-1)

            # Apply Gumbel-Softmax
            _logprobs = gumbel_softmax_sample(logprobs, temperature)
            _, it = torch.max(_logprobs.data, 1)
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))  # Gather the logprobs at sampled positions

        else:  # Various top-k and nucleus sampling methods
            logprobs = logprobs / temperature

            if sample_method.startswith('top'):  # Top-k or nucleus sampling
                top_num = float(sample_method[3:])

                if 0 < top_num < 1:  # Nucleus sampling (top-p)
                    # Implement nucleus sampling from "The Curious Case of Neural Text Degeneration"
                    probs = F.softmax(logprobs, dim=1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                    _cumsum = sorted_probs.cumsum(1)
                    mask = _cumsum < top_num
                    mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], 1)
                    sorted_probs = sorted_probs * mask.float()
                    sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
                    logprobs.scatter_(1, sorted_indices, sorted_probs.log())

                else:  # Standard top-k sampling
                    the_k = int(top_num)
                    tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                    topk, indices = torch.topk(logprobs, the_k, dim=1)
                    tmp = tmp.scatter(1, indices, topk)
                    logprobs = tmp

            # Sample from the modified distribution
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))  # Gather the logprobs at sampled positions

        return it, sampleLogprobs