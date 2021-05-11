# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from ctp.kernels import BaseKernel
from ctp.clutrr.models.util import lookup, uniform

from typing import List, Tuple, Optional

import logging

logger = logging.getLogger(__name__)

# Neural KB
class BatchNeuralKB(nn.Module):
    def __init__(self,
                 kernel: BaseKernel,
                 scoring_type: str = 'concat'):
        super().__init__()

        self.kernel = kernel
        self.scoring_type = scoring_type
        assert self.scoring_type in {'concat'}

    # Get score of ground relation from knowledge base of facts
    def score(self,
              rel: Tensor, arg1: Tensor, arg2: Tensor,  # Binary predicate, first entity, second entity
              facts: List[Tensor],                      # List of lists of facts (different facts for each batch)
              nb_facts: Tensor,                         # Number of facts for each example in the batch
              entity_embeddings: Optional[Tensor] = None,
              nb_entities: Optional[Tensor] = None) -> Tensor:
        # Returns -> tensor across batch of scores

        # [B, F, 3E] - batch, fact, 3 embeddings
        facts_emb = torch.cat(facts, dim=2)  # Get embeddings of facts

        # [B, 3E]
        batch_emb = torch.cat([rel, arg1, arg2], dim=1)  # Embedding of predicate for scoring

        # [B, F]
        batch_fact_scores = lookup(batch_emb, facts_emb, nb_facts, self.kernel)  # Matching-score for each fact

        # [B]
        res, _ = torch.max(batch_fact_scores, dim=1)  # Score predicate as max matching-score
        return res

    # Get score of relation that might contain variables
    def forward(self,
                rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],        # 'None' entity == variable
                facts: List[Tensor],                                                # List of lists of facts
                nb_facts: Tensor,                                                   # Number of facts
                entity_embeddings: Tensor,                                          # Entity embeddings
                nb_entities: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:  # Number of entities
        # Returns -> (score for first arg, score for second arg)

        # rel: [B, E], arg1: [B, E], arg2: [B, E]
        # facts: [B, F, E]
        # entity_embeddings: [B, N, E] (XXX: need no. entities)

        # [B, F, 3E]
        fact_emb = torch.cat(facts, dim=2)  # Get fact embeddings from list of facts

        fact_emb, nb_facts = uniform(rel, fact_emb, nb_facts)
        entity_embeddings, nb_entities = uniform(rel, entity_embeddings, nb_entities)

        batch_size = rel.shape[0]
        embedding_size = rel.shape[1]
        entity_size = entity_embeddings.shape[1]
        fact_size = fact_emb.shape[1]

        # [B, N, F, 3E] - batch, entity, fact, 3 embeddings
        # Repeat facts across entity embeddings
        fact_bnf3e = fact_emb.view(batch_size, 1, fact_size, -1).repeat(1, entity_size, 1, 1)

        # [B, N, F, E]
        # Repeat relation across entity embeddings and facts
        rel_bnfe = rel.view(batch_size, 1, 1, embedding_size).repeat(1, entity_size, fact_size, 1)

        # [B, N, F, E]
        # Repeat entity embeddings across facts
        emb_bnfe = entity_embeddings.view(batch_size, entity_size, 1, embedding_size).repeat(1, 1, fact_size, 1)

        # [B, F]
        fact_mask = torch.arange(fact_size, device=nb_facts.device)\
            .expand(batch_size, fact_size) < nb_facts.unsqueeze(1)
        # [B, N]
        entity_mask = torch.arange(entity_size, device=nb_entities.device)\
            .expand(batch_size, entity_size) < nb_entities.unsqueeze(1)

        # [B, N, F]
        mask = fact_mask.view(batch_size, 1, fact_size).repeat(1, entity_size, 1) * \
            entity_mask.view(batch_size, entity_size, 1).repeat(1, 1, fact_size)

        score_sp = score_po = None  # Score each arg as 'None' by default

        if arg1 is not None:
            # [B, N, F, E]
            # Repeat arg1 across entity embeddings and facts
            arg1_bnfe = arg1.view(batch_size, 1, 1, embedding_size).repeat(1, entity_size, fact_size, 1)

            # [B, N, F, 3E]
            # Construct query from relation, arg1, and entity embeddings
            query_bnf3e = torch.cat([rel_bnfe, arg1_bnfe, emb_bnfe], dim=3)

            # [B, N, F]
            # Compare query to facts using kernel
            scores_bnf = self.kernel(query_bnf3e, fact_bnf3e).view(batch_size, entity_size, fact_size)
            scores_bnf = scores_bnf * mask

            # [B, N]
            # Score arg1 with maximum score across entity embeddings
            score_sp, _ = torch.max(scores_bnf, dim=2)

        if arg2 is not None:
            # [B, N, F, E]
            # Repeat arg2 across entity embeddings and facts
            arg2_bnfe = arg2.view(batch_size, 1, 1, embedding_size).repeat(1, entity_size, fact_size, 1)

            # [B, N, F, 3E]
            # Construct query from relation, arg2, and entity embeddings
            query_bnf3e = torch.cat([rel_bnfe, emb_bnfe, arg2_bnfe], dim=3)

            # [B, N, F]
            # Compare query to facts using kernel
            scores_bnf = self.kernel(query_bnf3e, fact_bnf3e).view(batch_size, entity_size, fact_size)
            scores_bnf = scores_bnf * mask

            # [B, N]
            # Score arg2 with maximum score across entity embeddings
            score_po, _ = torch.max(scores_bnf, dim=2)

        return score_sp, score_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return embedding_vector
