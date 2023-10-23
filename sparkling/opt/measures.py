from enum import Enum
from typing import Optional

from sparkling.data.dataframe import SparklingDF


class Internal(Enum):
    """
    Enumeration of available internal measures (cluster validity indices).
    True flag indicates increasing function (e.g., bigger value means better clustering partition),
    while False indicates decreasing CVI (e.g., smaller value means better clustering result)
    """

    CALINSKI_HARABASZ = ('ch', False)
    DAVIES_BOULDIN = ('db', True)
    DAVIES_BOULDIN_ALTER = ('db*', True)
    DUNN = ('dunn', False)
    GD31 = ('gd31', False)
    GD33 = ('gd33', False)
    GD41 = ('gd41', False)
    GD41_APPROX = ('gd41*', False)
    GD43 = ('gd43', False)
    GD51 = ('gd51', False)
    GD51_APPROX = ('gd51*', False)
    GD53 = ('gd53', False)
    SILHOUETTE = ('sil', False)
    SILHOUETTE_APPROX = ('sil*', False)
    SF = ('sf', False)

    def evaluate(self, sparkling_df: SparklingDF, minimise: Optional[bool] = None) -> float:
        """
        Computes internal measure value for :class:`SparklingDF` using invoker's algorithm.

        >>> from sparkling.opt.measures import Internal
        >>> Internal.SILHOUETTE.evaluate(sparkling_df)  # 0.5

        :param sparkling_df: :class:`SparklingDF` with label column.
        :param minimise: specifies function monotonicity, e.g. if CVI is increasing,
        then minimise=True will invert the sign of a result. None returns original value.
        """

        name, decreasing = self.value
        jdf, dist = sparkling_df.jdf, sparkling_df.dist
        package = sparkling_df.jvm.ru.ifmo.rain.measures
        result = package.Internals.evaluate(name, jdf, dist)
        if minimise is None:
            return result
        sign = -1.0 if minimise ^ decreasing else 1.0
        return sign * result


class External(Enum):
    """
    Enumeration of available external measures.
    """

    RAND = ('rand', 'pairwise')
    JACCARD = ('jaccard', 'pairwise')
    FOWLKES_MALLOWS = ('fowlkesMallows', 'pairwise')
    PHI = ('phi', 'pairwise')

    class Pairwise:
        """
        Gathers external measures values, which are evaluated via pairwise
        comparison of each dataset's object's external class and computed label.
        """

        def __init__(self, rand, jaccard, fowlkes_mallows, phi):
            self.rand = rand
            self.jaccard = jaccard
            self.fowlkes_mallows = fowlkes_mallows
            self.phi = phi

    F1 = ('f1', 'conjugate')
    PURITY = ('purity', 'conjugate')
    ENTROPY = ('entropy', 'conjugate')
    MINKOWSKI = ('minkowski', 'conjugate')
    ADJUSTED_RAND = ('adjustedRand', 'conjugate')
    GOODMAN_KRUSKAL = ('goodmanKruskal', 'conjugate')
    VAR_INFO = ('varInformation', 'conjugate')

    class Conjugate:
        """
        Gathers external measures values, which are evaluated via contingency table.
        """

        def __init__(self, f1, purity, entropy, minkowski, adjusted_rand, goodman_kruskal, var_info):
            self.f1 = f1
            self.purity = purity
            self.entropy = entropy
            self.minkowski = minkowski
            self.adjusted_rand = adjusted_rand
            self.goodman_kruskal = goodman_kruskal
            self.var_info = var_info

    def evaluate(self, sparkling_df: SparklingDF) -> float:
        """
        Computes according external measure value for :class:`SparklingDF`.

        >>> from sparkling.opt.measures import External
        >>> External.F1.evaluate(sparkling_df)  # 0.7

        :param sparkling_df: :class:`SparklingDF` with both class and label column.
        """

        name, fun = self.value
        jvm_obj = getattr(External, f'_jvm_{fun}').__call__(sparkling_df)
        return getattr(jvm_obj, name).__call__()

    @staticmethod
    def pairwise(sparkling_df: SparklingDF) -> Pairwise:
        """
        Computes all pairwise (:class:`External.Pairwise`) external measure values for :class:`SparklingDF`.

        >>> from sparkling.opt.measures import External
        >>> External.pairwise(sparkling_df)  # Pairwise(rand=0.4, jaccard=0.5, ...)

        :param sparkling_df: :class:`SparklingDF` with both class and label column.
        """

        jvm_obj = External._jvm_pairwise(sparkling_df.jdf)
        return External.Pairwise(
            rand=jvm_obj.rand(),
            jaccard=jvm_obj.jaccard(),
            fowlkes_mallows=jvm_obj.fowlkesMallows(),
            phi=jvm_obj.phi()
        )

    @staticmethod
    def conjugate(sparkling_df: SparklingDF) -> Conjugate:
        """
        Computes all conjugate (:class:`External.Conjugate`) external measure values for :class:`SparklingDF`.

        >>> from sparkling.opt.measures import External
        >>> External.conjugate(sparkling_df)  # Conjugate(f1=0.7, purity=0.9, ...)

        :param sparkling_df: :class:`SparklingDF` with both class and label column.
        """

        jvm_obj = External._jvm_conjugate(sparkling_df.jdf)
        return External.Conjugate(
            f1=jvm_obj.f1(),
            purity=jvm_obj.purity(),
            entropy=jvm_obj.entropy(),
            minkowski=jvm_obj.minkowski(),
            adjusted_rand=jvm_obj.adjustedRand(),
            goodman_kruskal=jvm_obj.goodmanKruskal(),
            var_info=jvm_obj.varInformation()
        )

    @staticmethod
    def _jvm_pairwise(sparkling_df: SparklingDF):
        return sparkling_df.jvm.ru.ifmo.rain.measures.Externals.pairwise(sparkling_df.jdf)

    @staticmethod
    def _jvm_conjugate(sparkling_df: SparklingDF):
        return sparkling_df.jvm.ru.ifmo.rain.measures.Externals.conjugate(sparkling_df.jdf)
