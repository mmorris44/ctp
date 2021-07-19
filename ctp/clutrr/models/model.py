# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ctp.clutrr.models.kb import BatchNeuralKB
from ctp.clutrr.models.util import uniform

from ctp.reformulators import BaseReformulator
from ctp.reformulators import GNTPReformulator

from ctp.reinforcement.reinforce import ReinforceModule

from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)

# Encoder model
class BatchHoppy(nn.Module):
    def __init__(self,
                 model: BatchNeuralKB,                              # Neural KB
                 hops_lst: List[Tuple[BaseReformulator, bool]],     # List of reformulators
                 # Reformulator takes queries and reformulates into sub-queries
                 # Each reformulator learns to reformulate all queries, but each only in one way
                 k: int = 10,                                       # k-max parameter. Select top k scores in r_hop()
                 depth: int = 0,                                    # Depth to search before stopping
                 tnorm_name: str = 'min',                           # How to calculate scores as a conjunction
                 reinforce_module: ReinforceModule = None,          # Module containing REINFORCE
                 R: Optional[int] = None):                          # GNTP-R parameter from args?
        super().__init__()

        self.model: BatchNeuralKB = model
        self.k = k

        self.depth = depth
        assert self.depth >= 0

        self.tnorm_name = tnorm_name
        assert self.tnorm_name in {'min', 'prod', 'mean'}

        self.R = R

        self._hops_lst = nn.ModuleList([hops for hops, _ in hops_lst])
        self.hops_lst = hops_lst

        self.reinforce_module = reinforce_module

        logger.info(f'BatchHoppy(k={k}, depth={depth}, hops_lst={[h.__class__.__name__ for h in self._hops_lst]})')

    # Conjunction of relations
    def _tnorm(self, x: Tensor, y: Tensor) -> Tensor:
        res = None
        if self.tnorm_name == 'min':  # Default to use
            res = torch.min(x, y)
        elif self.tnorm_name == 'prod':
            res = x * y
        elif self.tnorm_name == 'mean':
            res = (x + y) / 2
        assert res is not None
        return res

    # Recursively expand relation
    def r_hop(self,
              rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],  # Predicate, first entity, second entity
              facts: List[Tensor],                      # List of lists of facts (different facts for each batch)
              nb_facts: Tensor,                         # Number of facts for each example in the batch
              entity_embeddings: Tensor,                # Entity embeddings
              nb_entities: Tensor,                      # Number of entities
              depth: int) -> Tuple[Tensor, Tensor]:
        assert (arg1 is None) ^ (arg2 is None)  # XOR. Exactly one variable
        assert depth >= 0

        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        # [B, N] - Batch, entity
        # Get scores from expanding recursively (scores for arg1, arg2 respectively)
        scores_sp, scores_po = self.r_forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=depth)
        scores = scores_sp if arg2 is None else scores_po

        k = min(self.k, scores.shape[1])  # Ensure k within bounds of tensor

        # [B, K], [B, K]
        z_scores, z_indices = torch.topk(scores, k=k, dim=1)  # Get top k scores

        dim_1 = torch.arange(z_scores.shape[0], device=z_scores.device).view(-1, 1).repeat(1, k).view(-1)
        dim_2 = z_indices.view(-1)

        # Get entity embeddings for top k scores
        entity_embeddings, _ = uniform(z_scores, entity_embeddings)

        z_emb = entity_embeddings[dim_1, dim_2].view(z_scores.shape[0], k, -1)

        assert z_emb.shape[0] == batch_size
        assert z_emb.shape[2] == embedding_size

        # Return top k scores and corresponding entities
        return z_scores, z_emb

    # score() is for p(s,o) ground predicates
    # forward() is for p(s, X) or p(X, o)

    # Get score of relation using max depth of model
    def score(self,
              rel: Tensor, arg1: Tensor, arg2: Tensor,
              facts: List[Tensor],
              nb_facts: Tensor,
              entity_embeddings: Tensor,
              nb_entities: Tensor) -> Tensor:
        res = self.r_score(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=self.depth)
        return res

    # Get max score of relation for up to given depth
    def r_score(self,
                rel: Tensor, arg1: Tensor, arg2: Tensor,
                facts: List[Tensor],
                nb_facts: Tensor,
                entity_embeddings: Tensor,
                nb_entities: Tensor,
                depth: int) -> Tensor:
        res = None
        for d in range(depth + 1):  # Check up to max depth
            if self.reinforce_module.use_rl:
                scores = self.depth_r_score_select_first_element(
                    rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=d)
            else:
                scores = self.depth_r_score(
                    rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=d)
            res = scores if res is None else torch.max(res, scores)  # Maximize score across depth
        return res

    # Get score of relation for given depth
    def depth_r_score_select_first_element(self,
                      rel: Tensor, arg1: Tensor, arg2: Tensor,  # Predicate, first entity, second entity
                      facts: List[Tensor],  # List of lists of facts (different facts for each batch)
                      nb_facts: Tensor,  # Number of facts for each example in the batch
                      entity_embeddings: Tensor,  # Entity embeddings
                      nb_entities: Tensor,  # Number of entities
                      depth: int) -> Tensor:  # How many more steps down before calling to KB
        assert depth >= 0

        if depth == 0:  # If depth reached, call to neural KB for score
            return self.model.score(rel, arg1, arg2,
                                    facts=facts, nb_facts=nb_facts,
                                    entity_embeddings=entity_embeddings, nb_entities=nb_entities)

        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        global_res = None

        mask = None

        new_hops_lst = self.hops_lst

        # [B, 3E] - B seems to be 576 with a batch size of 32 passed as a parameter
        state = torch.cat([rel, arg1, arg2], dim=1)[0:1]  # Embedding of first predicate in batch
        actions, action_counts = self.reinforce_module.get_actions(state)  # Get actions for first batch element
        actions = actions[0]

        # Iterate through reformulators (hops_generator is reformulator)
        # is_reversed decides if the next sub-goal is in the form p(a, X) or p(X, a)
        for rule_idx, (hops_generator, is_reversed) in enumerate(new_hops_lst):
            action = rule_idx

            # Skip reformulator if not chosen by selection module
            if action not in actions:
                continue

            sources, scores = arg1, None

            # XXX - IGNORE THIS FOR NOW
            prior = hops_generator.prior(rel)
            if prior is not None:  # Get a prior on the scores

                if mask is not None:
                    prior = prior * mask[:, rule_idx]
                    if (prior != 0.0).sum() == 0:
                        continue

                scores = prior

            hop_rel_lst = hops_generator(rel)  # Generate hops from relation (using the reformulator)
            nb_hops = len(hop_rel_lst)

            # For each hop in the hops to consider for the relation
            for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                # [B * S, E]
                sources_2d = sources.view(-1, embedding_size)
                nb_sources = sources_2d.shape[0]

                nb_branches = nb_sources // batch_size

                hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                if hop_idx < nb_hops:  # Scores are T-normed in one by one
                    # [B * S, K], [B * S, K, E]
                    if is_reversed:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                     facts, nb_facts, entity_embeddings, nb_entities,
                                                     depth=depth - 1)
                    else:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                     facts, nb_facts, entity_embeddings, nb_entities,
                                                     depth=depth - 1)
                    k = z_emb.shape[1]

                    # [B * S * K]
                    z_scores_1d = z_scores.view(-1)
                    # [B * S * K, E]
                    z_emb_2d = z_emb.view(-1, embedding_size)

                    # [B * S * K, E]
                    sources = z_emb_2d
                    # [B * S * K]
                    scores = z_scores_1d if scores is None \
                        else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                else:  # Final hop
                    # [B, S, E]
                    arg2_3d = arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    # [B * S, E]
                    arg2_2d = arg2_3d.view(-1, embedding_size)

                    # [B * S]
                    if is_reversed:
                        z_scores_1d = self.r_score(hop_rel_2d, arg2_2d, sources_2d,
                                                   facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                    else:
                        z_scores_1d = self.r_score(hop_rel_2d, sources_2d, arg2_2d,
                                                   facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)

                    scores = z_scores_1d if scores is None else self._tnorm(z_scores_1d, scores)

            if scores is not None:
                scores_2d = scores.view(batch_size, -1)
                res, _ = torch.max(scores_2d, dim=1)
            else:
                res = self.model.score(rel, arg1, arg2,
                                       facts=facts, nb_facts=nb_facts,
                                       entity_embeddings=entity_embeddings, nb_entities=nb_entities)

            # Maximize score across reformulators
            global_res = res if global_res is None else torch.max(global_res, res)

            # Apply reward
            if self.reinforce_module.mode == 'train':
                state_batch: Tensor = state.expand(batch_size, -1)
                action_batch: Tensor = torch.tensor([action]).expand(batch_size)
                reward_batch: Tensor = res
                self.reinforce_module.apply_reward(state_batch, action_batch, reward_batch)

        return global_res

    # Get score of relation for given depth
    def depth_r_score_select_tensor_slicing(self,
                             rel: Tensor, arg1: Tensor, arg2: Tensor,  # Predicate, first entity, second entity
                             facts: List[Tensor],  # List of lists of facts (different facts for each batch)
                             nb_facts: Tensor,  # Number of facts for each example in the batch
                             entity_embeddings: Tensor,  # Entity embeddings
                             nb_entities: Tensor,  # Number of entities
                             depth: int) -> Tensor:  # How many more steps down before calling to KB
        assert depth >= 0

        if depth == 0:  # If depth reached, call to neural KB for score
            return self.model.score(rel, arg1, arg2,
                                    facts=facts, nb_facts=nb_facts,
                                    entity_embeddings=entity_embeddings, nb_entities=nb_entities)

        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        global_res = None

        mask = None

        new_hops_lst = self.hops_lst

        # Repeat arguments if mismatching batch sizes
        if rel.shape[0] > nb_facts.shape[0]:
            expansion_factor = rel.shape[0] // nb_facts.shape[0]
            facts = [facts[0].repeat(expansion_factor, 1, 1), facts[1].repeat(expansion_factor, 1, 1),
                     facts[2].repeat(expansion_factor, 1, 1)]
            nb_facts = nb_facts.repeat(expansion_factor)
            entity_embeddings = entity_embeddings.repeat(expansion_factor, 1, 1)
            nb_entities = nb_entities.repeat(expansion_factor)

        # [B, 3E] - B seems to be 576 with a batch size of 32 passed as a parameter
        state = torch.cat([rel, arg1, arg2], dim=1)  # Embedding of predicate
        actions, action_counts = self.reinforce_module.get_actions(state)  # Get actions for each batch element

        # Iterate through reformulators (hops_generator is reformulator)
        # is_reversed decides if the next sub-goal is in the form p(a, X) or p(X, a)
        for rule_idx, (hops_generator, is_reversed) in enumerate(new_hops_lst):
            action = rule_idx

            # Restrict arguments to only include batch elements that have been selected for this reformulator
            selected_rel = torch.zeros(action_counts[action], rel.shape[1])
            selected_arg1 = torch.zeros(action_counts[action], arg1.shape[1])
            selected_arg2 = torch.zeros(action_counts[action], arg2.shape[1])
            selected_facts0 = torch.zeros(action_counts[action], facts[0].shape[1], facts[0].shape[2])
            selected_facts1 = torch.zeros(action_counts[action], facts[1].shape[1], facts[1].shape[2])
            selected_facts2 = torch.zeros(action_counts[action], facts[2].shape[1], facts[2].shape[2])
            selected_nb_facts = torch.zeros(action_counts[action])
            selected_entity_embeddings = torch.zeros(action_counts[action], entity_embeddings.shape[1],
                                                     entity_embeddings.shape[2])
            selected_nb_entities = torch.zeros(action_counts[action])

            action_number = 0
            for i in range(batch_size):
                if action in actions[i]:
                    selected_rel[action_number] = rel[i]
                    selected_arg1[action_number] = arg1[i]
                    selected_arg2[action_number] = arg2[i]
                    selected_facts0[action_number] = facts[0][i]
                    selected_facts1[action_number] = facts[1][i]
                    selected_facts2[action_number] = facts[2][i]
                    selected_nb_facts[action_number] = nb_facts[i]
                    selected_entity_embeddings[action_number] = entity_embeddings[i]
                    selected_nb_entities[action_number] = nb_entities[i]
                    action_number += 1
            selected_facts = [selected_facts0, selected_facts1, selected_facts2]

            sources, scores = selected_arg1, None

            # XXX - IGNORE THIS FOR NOW
            prior = hops_generator.prior(selected_rel)
            if prior is not None:  # Get a prior on the scores

                if mask is not None:
                    prior = prior * mask[:, rule_idx]
                    if (prior != 0.0).sum() == 0:
                        continue

                scores = prior

            hop_rel_lst = hops_generator(selected_rel)  # Generate hops from relation (using the reformulator)
            nb_hops = len(hop_rel_lst)

            # For each hop in the hops to consider for the relation
            for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                # [B * S, E]
                sources_2d = sources.view(-1, embedding_size)
                nb_sources = sources_2d.shape[0]

                # nb_branches = nb_sources // batch_size
                nb_branches = nb_sources // action_counts[action]

                hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                if hop_idx < nb_hops:  # Scores are T-normed in one by one
                    # [B * S, K], [B * S, K, E]
                    if is_reversed:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                     selected_facts, selected_nb_facts, selected_entity_embeddings,
                                                     selected_nb_entities, depth=depth - 1)
                    else:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                     selected_facts, selected_nb_facts, selected_entity_embeddings,
                                                     selected_nb_entities, depth=depth - 1)
                    k = z_emb.shape[1]

                    # [B * S * K]
                    z_scores_1d = z_scores.view(-1)
                    # [B * S * K, E]
                    z_emb_2d = z_emb.view(-1, embedding_size)

                    # [B * S * K, E]
                    sources = z_emb_2d
                    # [B * S * K]
                    scores = z_scores_1d if scores is None \
                        else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                else:  # Final hop
                    # [B, S, E]
                    arg2_3d = selected_arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    # [B * S, E]
                    arg2_2d = arg2_3d.view(-1, embedding_size)

                    # [B * S]
                    if is_reversed:
                        z_scores_1d = self.r_score(hop_rel_2d, arg2_2d, sources_2d,
                                                   selected_facts, selected_nb_facts, selected_entity_embeddings,
                                                   selected_nb_entities, depth=depth - 1)
                    else:
                        z_scores_1d = self.r_score(hop_rel_2d, sources_2d, arg2_2d,
                                                   selected_facts, selected_nb_facts, selected_entity_embeddings,
                                                   selected_nb_entities, depth=depth - 1)

                    scores = z_scores_1d if scores is None else self._tnorm(z_scores_1d, scores)

            if scores is not None:
                # scores_2d = scores.view(batch_size, -1)
                scores_2d = scores.view(action_counts[action], -1)
                selected_res, _ = torch.max(scores_2d, dim=1)

                # Add scores of zero if reformulator was not selected - to match the batch sizes
                res = torch.zeros(batch_size)
                action_number = 0
                for i in range(batch_size):
                    if action in actions[i]:
                        res[i] = selected_res[action_number]
                        action_number += 1
            else:
                res = self.model.score(rel, arg1, arg2,
                                       facts=facts, nb_facts=nb_facts,
                                       entity_embeddings=entity_embeddings, nb_entities=nb_entities)

            # Maximize score across reformulators
            global_res = res if global_res is None else torch.max(global_res, res)

        return global_res

    # Get score using reformulator selector - this is HORRIBLY SLOW
    def depth_r_score_select_recursive(self,
                             rel: Tensor, arg1: Tensor, arg2: Tensor,  # Predicate, first entity, second entity
                             facts: List[Tensor],          # List of lists of facts (different facts for each batch)
                             nb_facts: Tensor,             # Number of facts for each example in the batch
                             entity_embeddings: Tensor,    # Entity embeddings
                             nb_entities: Tensor,          # Number of entities
                             depth: int) -> Tensor:        # How many more steps down before calling to KB
        assert depth >= 0

        if depth == 0:  # If depth reached, call to neural KB for score
            return self.model.score(rel, arg1, arg2,
                                    facts=facts, nb_facts=nb_facts,
                                    entity_embeddings=entity_embeddings, nb_entities=nb_entities)

        batch_size, embedding_size = nb_facts.shape[0], rel.shape[1]  # Not rel.shape[0], to avoid batch size bug
        global_res = None
        mask = None

        if batch_size > 1:  # If more than 1 element in the batch
            # print("-----")
            # print("before un-batch:", rel.size(), arg1.size(), arg2.size(), facts[0].size(),
            #       facts[1].size(), facts[2].size(), nb_facts.size(),
            #       entity_embeddings.size(), nb_entities.size(), depth)

            # Un-batch, call, then combine together again
            batch_res = torch.zeros(batch_size)
            for i in range(batch_size):
                score = self.depth_r_score_select(rel=rel[i:i+1], arg1=arg1[i:i+1], arg2=arg2[i:i+1],
                                                  facts=[x[i:i+1] for x in facts], nb_facts=nb_facts[i:i+1],
                                                  entity_embeddings=entity_embeddings[i:i+1],
                                                  nb_entities=nb_entities[i:i+1], depth=depth)
                batch_res[i] = score.item()
            print("Batch res:", batch_res)
            return batch_res
        # Can assume batch size is 1 from here onwards

        if rel.shape[0] > 1:  # If more relations than other parameters in the batch (bug)
            rel = rel[0:1]
            arg1 = arg1[0:1]
            arg2 = arg2[0:1]

        # print("depth_r_score_select:", rel.size(), arg1.size(), arg2.size(), facts[0].size(),
        #       facts[1].size(), facts[2].size(), nb_facts.size(),
        #       entity_embeddings.size(), nb_entities.size(), depth)

        new_hops_lst = self.hops_lst

        # [B, 3E] - B seems to be 576 with a batch size of 32 passed as a parameter
        state = torch.cat([rel, arg1, arg2], dim=1)[0]  # Embedding of predicate

        actions = self.reinforce_module.get_actions(state)  # Get actions

        # Iterate through reformulators (hops_generator is reformulator)
        # is_reversed decides if the next sub-goal is in the form p(a, X) or p(X, a)
        for rule_idx, (hops_generator, is_reversed) in enumerate(new_hops_lst):
            if rule_idx not in actions:
                continue  # Skip reformulators not chosen by select module

            sources, scores = arg1, None

            # XXX - IGNORE THIS FOR NOW
            prior = hops_generator.prior(rel)
            if prior is not None:  # Get a prior on the scores

                if mask is not None:
                    prior = prior * mask[:, rule_idx]
                    if (prior != 0.0).sum() == 0:
                        continue

                scores = prior

            hop_rel_lst = hops_generator(rel)  # Generate hops from relation (using the reformulator)
            nb_hops = len(hop_rel_lst)

            # For each hop in the hops to consider for the relation
            for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                # [B * S, E]
                sources_2d = sources.view(-1, embedding_size)
                nb_sources = sources_2d.shape[0]

                nb_branches = nb_sources // batch_size

                hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                if hop_idx < nb_hops:  # Scores are T-normed in one by one
                    # [B * S, K], [B * S, K, E]
                    if is_reversed:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                     facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                    else:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                     facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                    k = z_emb.shape[1]

                    # [B * S * K]
                    z_scores_1d = z_scores.view(-1)
                    # [B * S * K, E]
                    z_emb_2d = z_emb.view(-1, embedding_size)

                    # [B * S * K, E]
                    sources = z_emb_2d
                    # [B * S * K]
                    scores = z_scores_1d if scores is None \
                        else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                else:  # Final hop
                    # [B, S, E]
                    arg2_3d = arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    # [B * S, E]
                    arg2_2d = arg2_3d.view(-1, embedding_size)

                    # [B * S]
                    if is_reversed:
                        z_scores_1d = self.r_score(hop_rel_2d, arg2_2d, sources_2d,
                                                   facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                    else:
                        z_scores_1d = self.r_score(hop_rel_2d, sources_2d, arg2_2d,
                                                   facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)

                    scores = z_scores_1d if scores is None else self._tnorm(z_scores_1d, scores)

            if scores is not None:
                scores_2d = scores.view(batch_size, -1)
                res, _ = torch.max(scores_2d, dim=1)
            else:
                res = self.model.score(rel, arg1, arg2,
                                       facts=facts, nb_facts=nb_facts,
                                       entity_embeddings=entity_embeddings, nb_entities=nb_entities)

            # Apply reward to selection module
            # print("Applying reward:", state, torch.tensor(rule_idx), res)
            self.reinforce_module.apply_reward(state=state, action=torch.tensor(rule_idx), reward=res)

            # Maximize score across reformulators
            global_res = res if global_res is None else torch.max(global_res, res)

        return global_res

    # Get score of relation for given depth
    def depth_r_score(self,
                      rel: Tensor, arg1: Tensor, arg2: Tensor,  # Predicate, first entity, second entity
                      facts: List[Tensor],          # List of lists of facts (different facts for each batch)
                      nb_facts: Tensor,             # Number of facts for each example in the batch
                      entity_embeddings: Tensor,    # Entity embeddings
                      nb_entities: Tensor,          # Number of entities
                      depth: int) -> Tensor:        # How many more steps down before calling to KB
        assert depth >= 0

        if depth == 0:  # If depth reached, call to neural KB for score
            return self.model.score(rel, arg1, arg2,
                                    facts=facts, nb_facts=nb_facts,
                                    entity_embeddings=entity_embeddings, nb_entities=nb_entities)

        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        global_res = None

        mask = None

        new_hops_lst = self.hops_lst

        if self.R is not None:  # If GNTP reformulators included? ---> IGNORE THIS BLOCK OF CODE
            batch_rules_scores = torch.cat([h.prior(rel).view(-1, 1) for h, _ in self.hops_lst], 1)
            topk, indices = torch.topk(batch_rules_scores, self.R)

            # [R x E]
            rule_heads = torch.cat([h.head for h, _ in self.hops_lst], dim=0)
            rule_body1s = torch.cat([h.memory_lst[0] for h, _ in self.hops_lst], dim=0)
            rule_body2s = torch.cat([h.memory_lst[1] for h, _ in self.hops_lst], dim=0)

            kernel = self.hops_lst[0][0].kernel
            new_rule_heads = F.embedding(indices, rule_heads)
            new_rule_body1s = F.embedding(indices, rule_body1s)
            new_rule_body2s = F.embedding(indices, rule_body2s)

            # print(new_rule_heads.shape[1], self.R)
            assert new_rule_heads.shape[1] == self.R

            new_hops_lst = []
            for i in range(new_rule_heads.shape[1]):
                r = GNTPReformulator(kernel=kernel, head=new_rule_heads[:, i, :],  # Generates hops from relation?
                                     body=[new_rule_body1s[:, i, :], new_rule_body2s[:, i, :]])
                new_hops_lst += [(r, False)]

        # Iterate through reformulators (hops_generator is reformulator)
        # is_reversed decides if the next sub-goal is in the form p(a, X) or p(X, a)
        for rule_idx, (hops_generator, is_reversed) in enumerate(new_hops_lst):
            sources, scores = arg1, None

            # XXX - IGNORE THIS FOR NOW
            prior = hops_generator.prior(rel)
            if prior is not None:  # Get a prior on the scores

                if mask is not None:
                    prior = prior * mask[:, rule_idx]
                    if (prior != 0.0).sum() == 0:
                        continue

                scores = prior

            hop_rel_lst = hops_generator(rel)  # Generate hops from relation (using the reformulator)
            nb_hops = len(hop_rel_lst)

            # For each hop in the hops to consider for the relation
            for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                # [B * S, E]
                sources_2d = sources.view(-1, embedding_size)
                nb_sources = sources_2d.shape[0]

                nb_branches = nb_sources // batch_size

                hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                if hop_idx < nb_hops:  # Scores are T-normed in one by one
                    # [B * S, K], [B * S, K, E]
                    if is_reversed:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                     facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                    else:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                     facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                    k = z_emb.shape[1]

                    # [B * S * K]
                    z_scores_1d = z_scores.view(-1)
                    # [B * S * K, E]
                    z_emb_2d = z_emb.view(-1, embedding_size)

                    # [B * S * K, E]
                    sources = z_emb_2d
                    # [B * S * K]
                    scores = z_scores_1d if scores is None \
                        else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                else:  # Final hop
                    # [B, S, E]
                    arg2_3d = arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    # [B * S, E]
                    arg2_2d = arg2_3d.view(-1, embedding_size)

                    # [B * S]
                    if is_reversed:
                        z_scores_1d = self.r_score(hop_rel_2d, arg2_2d, sources_2d,
                                                   facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                    else:
                        z_scores_1d = self.r_score(hop_rel_2d, sources_2d, arg2_2d,
                                                   facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)

                    scores = z_scores_1d if scores is None else self._tnorm(z_scores_1d, scores)

            if scores is not None:
                scores_2d = scores.view(batch_size, -1)
                res, _ = torch.max(scores_2d, dim=1)
            else:
                res = self.model.score(rel, arg1, arg2,
                                       facts=facts, nb_facts=nb_facts,
                                       entity_embeddings=entity_embeddings, nb_entities=nb_entities)

            # Maximize score across reformulators
            global_res = res if global_res is None else torch.max(global_res, res)

        return global_res

    def forward(self,
                rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                facts: List[Tensor],
                nb_facts: Tensor,
                entity_embeddings: Tensor,
                nb_entities: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        res_sp, res_po = self.r_forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=self.depth)
        return res_sp, res_po

    def r_forward(self,
                  rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                  facts: List[Tensor],
                  nb_facts: Tensor,
                  entity_embeddings: Tensor,
                  nb_entities: Tensor,
                  depth: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        res_sp, res_po = None, None
        for d in range(depth + 1):
            if self.reinforce_module.use_rl:
                scores_sp, scores_po = self.depth_r_forward_select_first_element(rel, arg1, arg2, facts, nb_facts,
                                                                                 entity_embeddings, nb_entities,
                                                                                 depth=d)
            else:
                scores_sp, scores_po = self.depth_r_forward(rel, arg1, arg2, facts, nb_facts,
                                                            entity_embeddings, nb_entities, depth=d)
            res_sp = scores_sp if res_sp is None else torch.max(res_sp, scores_sp)
            res_po = scores_po if res_po is None else torch.max(res_po, scores_po)
        return res_sp, res_po

    def depth_r_forward_select_first_element(self,
                        rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                        facts: List[Tensor],
                        nb_facts: Tensor,
                        entity_embeddings: Tensor,
                        nb_entities: Tensor,
                        depth: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        if depth == 0:
            return self.model.forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities)

        global_scores_sp = global_scores_po = None

        mask = None
        new_hops_lst = self.hops_lst

        # Account for arg1 XOR arg2 being None by using a zero tensor instead
        arg1_state, arg2_state = arg1, arg2
        if arg1 is None:
            arg1_state = torch.zeros(rel.shape[0], rel.shape[1])
        if arg2 is None:
            arg2_state = torch.zeros(rel.shape[0], rel.shape[1])

        state = torch.cat([rel, arg1_state, arg2_state], dim=1)[0:1]  # Embedding of first predicate in batch
        actions, action_counts = self.reinforce_module.get_actions(state)  # Get actions for first batch element
        actions = actions[0]

        for rule_idx, (hop_generators, is_reversed) in enumerate(new_hops_lst):
            action = rule_idx

            # Skip reformulator if not chosen by selection module
            if action not in actions:
                continue

            scores_sp = scores_po = None
            hop_rel_lst = hop_generators(rel)
            nb_hops = len(hop_rel_lst)

            if arg1 is not None:
                sources, scores = arg1, None

                # XXX
                prior = hop_generators.prior(rel)
                if prior is not None:

                    if mask is not None:
                        prior = prior * mask[:, rule_idx]
                        if (prior != 0.0).sum() == 0:
                            continue

                    scores = prior

                for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                    # [B * S, E]
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]

                    nb_branches = nb_sources // batch_size

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:
                        # [B * S, K], [B * S, K, E]
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        k = z_emb.shape[1]

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, N]
                        if is_reversed:
                            _, scores_sp = self.r_forward(hop_rel_2d, None, sources_2d,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            scores_sp, _ = self.r_forward(hop_rel_2d, sources_2d, None,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)

                        nb_entities_ = scores_sp.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, nb_entities_)
                            scores_sp = self._tnorm(scores, scores_sp)

                            # [B, S, N]
                            scores_sp = scores_sp.view(batch_size, -1, nb_entities_)
                            # [B, N]
                            scores_sp, _ = torch.max(scores_sp, dim=1)

            if arg2 is not None:
                sources, scores = arg2, None

                # XXX
                prior = hop_generators.prior(rel)
                if prior is not None:
                    scores = prior
                # scores = hop_generators.prior(rel)

                for hop_idx, hop_rel in enumerate(reversed([h for h in hop_rel_lst]), start=1):
                    # [B * S, E]
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]

                    nb_branches = nb_sources // batch_size

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:
                        # [B * S, K], [B * S, K, E]
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        k = z_emb.shape[1]

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, N]
                        if is_reversed:
                            scores_po, _ = self.r_forward(hop_rel_2d, sources_2d, None,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            _, scores_po = self.r_forward(hop_rel_2d, None, sources_2d,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)

                        nb_entities_ = scores_po.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, nb_entities_)
                            scores_po = self._tnorm(scores, scores_po)

                            # [B, S, N]
                            scores_po = scores_po.view(batch_size, -1, nb_entities_)
                            # [B, N]
                            scores_po, _ = torch.max(scores_po, dim=1)

            if scores_sp is None and scores_po is None:
                scores_sp, scores_po = self.model.forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities)

            global_scores_sp = scores_sp if global_scores_sp is None else torch.max(global_scores_sp, scores_sp)
            global_scores_po = scores_po if global_scores_po is None else torch.max(global_scores_po, scores_po)

            # Apply reward
            if self.reinforce_module.mode == 'train':
                if scores_sp is not None:
                    state_batch: Tensor = state.expand(batch_size, -1)
                    action_batch: Tensor = torch.tensor([action]).expand(batch_size)
                    reward_batch, _ = torch.max(scores_sp, dim=1)  # Max reward across entities
                    self.reinforce_module.apply_reward(state_batch, action_batch, reward_batch)
                if scores_po is not None:
                    state_batch: Tensor = state.expand(batch_size, -1)
                    action_batch: Tensor = torch.tensor([action]).expand(batch_size)
                    reward_batch, _ = torch.max(scores_po, dim=1)  # Max reward across entities
                    self.reinforce_module.apply_reward(state_batch, action_batch, reward_batch)

        if global_scores_sp is None and global_scores_po is None:
            global_scores_sp, global_scores_po = self.model.forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities)

        return global_scores_sp, global_scores_po

    def depth_r_forward(self,
                        rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                        facts: List[Tensor],
                        nb_facts: Tensor,
                        entity_embeddings: Tensor,
                        nb_entities: Tensor,
                        depth: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        if depth == 0:
            return self.model.forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities)

        global_scores_sp = global_scores_po = None

        mask = None
        new_hops_lst = self.hops_lst

        if self.R is not None:
            batch_rules_scores = torch.cat([h.prior(rel).view(-1, 1) for h, _ in self.hops_lst], 1)
            topk, indices = torch.topk(batch_rules_scores, self.R)

            # [R x E]
            rule_heads = torch.cat([h.head for h, _ in self.hops_lst], dim=0)
            rule_body1s = torch.cat([h.memory_lst[0] for h, _ in self.hops_lst], dim=0)
            rule_body2s = torch.cat([h.memory_lst[1] for h, _ in self.hops_lst], dim=0)

            kernel = self.hops_lst[0][0].kernel
            new_rule_heads = F.embedding(indices, rule_heads)
            new_rule_body1s = F.embedding(indices, rule_body1s)
            new_rule_body2s = F.embedding(indices, rule_body2s)

            assert new_rule_heads.shape[1] == self.R

            new_hops_lst = []
            for i in range(new_rule_heads.shape[1]):
                r = GNTPReformulator(kernel=kernel, head=new_rule_heads[:, i, :],
                                     body=[new_rule_body1s[:, i, :], new_rule_body2s[:, i, :]])
                new_hops_lst += [(r, False)]

        for rule_idx, (hop_generators, is_reversed) in enumerate(new_hops_lst):
            scores_sp = scores_po = None
            hop_rel_lst = hop_generators(rel)
            nb_hops = len(hop_rel_lst)

            if arg1 is not None:
                sources, scores = arg1, None

                # XXX
                prior = hop_generators.prior(rel)
                if prior is not None:

                    if mask is not None:
                        prior = prior * mask[:, rule_idx]
                        if (prior != 0.0).sum() == 0:
                            continue

                    scores = prior

                for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                    # [B * S, E]
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]

                    nb_branches = nb_sources // batch_size

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:
                        # [B * S, K], [B * S, K, E]
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        k = z_emb.shape[1]

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, N]
                        if is_reversed:
                            _, scores_sp = self.r_forward(hop_rel_2d, None, sources_2d,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            scores_sp, _ = self.r_forward(hop_rel_2d, sources_2d, None,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)

                        nb_entities_ = scores_sp.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, nb_entities_)
                            scores_sp = self._tnorm(scores, scores_sp)

                            # [B, S, N]
                            scores_sp = scores_sp.view(batch_size, -1, nb_entities_)
                            # [B, N]
                            scores_sp, _ = torch.max(scores_sp, dim=1)

            if arg2 is not None:
                sources, scores = arg2, None

                # XXX
                prior = hop_generators.prior(rel)
                if prior is not None:
                    scores = prior
                # scores = hop_generators.prior(rel)

                for hop_idx, hop_rel in enumerate(reversed([h for h in hop_rel_lst]), start=1):
                    # [B * S, E]
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]

                    nb_branches = nb_sources // batch_size

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:
                        # [B * S, K], [B * S, K, E]
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                         facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        k = z_emb.shape[1]

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, N]
                        if is_reversed:
                            scores_po, _ = self.r_forward(hop_rel_2d, sources_2d, None,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)
                        else:
                            _, scores_po = self.r_forward(hop_rel_2d, None, sources_2d,
                                                          facts, nb_facts, entity_embeddings, nb_entities, depth=depth - 1)

                        nb_entities_ = scores_po.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, nb_entities_)
                            scores_po = self._tnorm(scores, scores_po)

                            # [B, S, N]
                            scores_po = scores_po.view(batch_size, -1, nb_entities_)
                            # [B, N]
                            scores_po, _ = torch.max(scores_po, dim=1)

            if scores_sp is None and scores_po is None:
                scores_sp, scores_po = self.model.forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities)

            global_scores_sp = scores_sp if global_scores_sp is None else torch.max(global_scores_sp, scores_sp)
            global_scores_po = scores_po if global_scores_po is None else torch.max(global_scores_po, scores_po)

        if global_scores_sp is None and global_scores_po is None:
            global_scores_sp, global_scores_po = self.model.forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities)

        return global_scores_sp, global_scores_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return self.model.factor(embedding_vector)

    def extra_factors(self,
                      rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor]) -> List[Tensor]:
        return [hop_generator(rel) for hop_generators in self.hops_lst for hop_generator in hop_generators]
