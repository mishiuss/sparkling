from enum import Enum
from typing import Optional


class Distance(Enum):
    """ Available metrics for intra modality distance """

    EUCLIDEAN = 'EUCLIDEAN'
    MANHATTAN = 'MANHATTAN'
    COSINE = 'COSINE'
    CHEBYSHEV = 'CHEBYSHEV'
    CANBERRA = 'CANBERRA'


class ModalInfo:
    """ Modality meta information """

    def __init__(
            self, name: str,
            dim: int = None, w_dim: Optional[int] = None,
            weight: float = None, metric: str = None, norm: float = None
    ):
        self.name, self.dim = name, dim
        self.weight, self.metric, self.norm = weight, metric, norm
        self.w_dim = w_dim if w_dim is not None else dim
