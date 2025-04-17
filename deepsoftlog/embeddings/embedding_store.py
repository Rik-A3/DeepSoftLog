import math
from typing import Iterable

import torch
from torch import nn

from ..parser.vocabulary import Vocabulary
from .distance import embedding_similarity
from .initialize_vector import Initializer
from ..logic.soft_term import TensorTerm, TextTerm
from .nn_models import EmbeddingFunctor
from deepsoftlog.algebraic_prover.terms.expression import Expr


class EmbeddingStore(nn.Module):
    """
    Stores the embeddings of soft constants,
    and the models for soft functors.
    """

    def __init__(self, ndim: int, initializer: Initializer, vocabulary: Vocabulary):
        super().__init__()
        self.ndim = ndim
        self.device = 'cpu'

        print("- Initializing embeddings with vocabulary:", vocabulary)
        self.constant_embeddings = nn.ParameterDict()
        for name in vocabulary.get_constants():
            self.constant_embeddings[name] = initializer(name)
        self.functor_embeddings = nn.ModuleDict()
        for signature in vocabulary.get_functors():
            self.functor_embeddings[str(signature)] = initializer(signature)
        self._cache = dict()

    def soft_unify_score(self, t1: Expr, t2: Expr, distance_metric: str):
        if distance_metric == "dummy":
            return math.log(0.6)

        sign = frozenset([t1, t2])
        if sign not in self._cache:
            e1, e2 = self.forward(t1), self.forward(t2)
            score = embedding_similarity(e1, e2, distance_metric)
            self._cache[sign] = score
        return self._cache[sign]

    def forward(self, term: Expr):
        assert term.get_predicate() != ("~", 1), \
            f"Cannot embed embedded term `{term}`."
        if term.get_arity() == 0:
            e = self._embed_constant(term)
        else:
            e = self._embed_functor(term)
        return e

    def _embed_constant(self, term: Expr):
        if isinstance(term, TensorTerm) or isinstance(term, TextTerm):
            return term.get_tensor().to(self.device)

        name = term.functor
        return self.constant_embeddings[name]

    def _embed_functor(self, functor: Expr):
        name = str(functor.get_predicate())
        embedded_args = [self(a) for a in functor.arguments]
        embedded_args = torch.stack(embedded_args)
        functor_model = self.functor_embeddings[name]

        embedding = functor_model(embedded_args)
        return embedding

    def clear_cache(self):
        self._cache = dict()

    def to(self, device):
        self.device = device
        return super().to(device)

    def get_soft_unification_matrix(self, distance_metric: str, names):
        n = len(names)
        matrix = torch.zeros(n, n)
        for i, c1 in enumerate(names):
            for j, c2 in enumerate(names):
                e1, e2 = self.constant_embeddings[c1], self.constant_embeddings[c2]
                matrix[i, j] = embedding_similarity(e1, e2, distance_metric) # log probabilities
        return matrix.detach().numpy()

def create_embedding_store(config, vocab_sources: Iterable) -> EmbeddingStore:
    ndim = config['embedding_dimensions']
    vocabulary = create_vocabulary(vocab_sources)
    initializer = Initializer(EmbeddingFunctor, config['embedding_initialization'], ndim, config.get("text_embedding_mode"))
    store = EmbeddingStore(ndim, initializer, vocabulary)
    return store


def create_vocabulary(vocab_sources: Iterable) -> Vocabulary:
    vocabulary = Vocabulary()
    for source in vocab_sources:
        vocabulary += source.get_vocabulary()
    return vocabulary
