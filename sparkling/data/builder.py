from typing import Set

import pandas as pd
import pyspark
import re
from pyspark.ml.linalg import VectorUDT as VectorType
from pyspark.mllib.linalg import VectorUDT as OldVectorType
from pyspark.sql.types import NumericType, StringType, IntegralType

from sparkling.emb.image_emb import ImageEmb
from sparkling.emb.tabular_emb import VectorColTabularEmb, MultiColTabularEmb
from sparkling.emb.text_emb import TextEmb
from sparkling.emb.tf import TFImages, TFTexts
from sparkling.emb.tf.tf_image import TFImageEmb
from sparkling.emb.tf.tf_text import TFTextEmb
from sparkling.emb.torch import TorchImages, TorchTexts
from sparkling.emb.torch.torch_image import TorchImageEmb
from sparkling.emb.torch.torch_text import TorchTextEmb
from .creation import *
from .dataframe import SparklingDF
from .modals import Distance
from .monad import DataMonad
from sparkling.util.logger import SparklingLogger


class SparklingBuilder:
    """
    Builds and runs pipeline to create a :class`SparklingDF`
    """

    DEFAULT_CONF = pyspark.SparkConf().setMaster('[*]').setAppName('sparkling')
    NAME_REGEX = re.compile('[a-z]\\w*', flags=re.ASCII)

    def __init__(
            self,
            dataframe: Union[pyspark.sql.DataFrame, pd.DataFrame], *,
            class_col: Optional[str] = None,
            partitions: Optional[Union[int, str]] = None
    ):
        """
        Instantiates builder with no modalities and raw dataframe

        :param dataframe: pyspark or pandas raw dataframe
        :param: class_col: name of the column that contains external labels (if present)
        :param: partitions: dataframe repartition mode.
        - None value means no repartition needed;
        - Integer value means the number of target partitions;
        - 'auto' performs repartition by some heuristic
        """

        self.sc = SparkContext.getOrCreate(self.DEFAULT_CONF)
        if isinstance(dataframe, pd.DataFrame):
            from .converter import PandasConverter
            dataframe = PandasConverter.convert(self.sc, dataframe)
        if not isinstance(dataframe, pyspark.sql.DataFrame):
            raise ValueError("Expected pandas.Dataframe or pyspark.sql.DataFrame")
        self._dtypes = {str(f.name): f.dataType for f in dataframe.schema.fields}
        self._df, self.class_col = dataframe, self._verify_class_col(class_col)

        self._embeddings, self._metrics, self._weights = dict(), dict(), dict()
        self._used_columns, self._partitions = set(), partitions

    def _verify_type(self, col_name, allowed_types):
        col_type = self._dtypes.get(col_name)
        if col_type is None:
            raise ValueError(f"Column '{col_name}' does not exist")
        if not isinstance(col_type, allowed_types):
            raise ValueError(f"Column '{col_name}' has incorrect type '{col_type}'. Expected one of {allowed_types}")

    def _verify_class_col(self, class_col):
        if class_col is not None:
            self._verify_type(class_col, (IntegralType, StringType))
        return class_col

    def _verify_name_usage(self, name):
        if self.NAME_REGEX.fullmatch(name) is None:
            raise ValueError(f"Name '{name}' does not match regex")
        self._verify_not_used(name)

    def _verify_not_used(self, name):
        if name in self._used_columns or name is self._embeddings.keys():
            raise ValueError(f"Name '{name}' has been already used")

    def _check_weights_undefined(self):
        if 0 < len(self._weights) < len([w for w in self._weights if w is None]):
            raise ValueError("You should either define weights for each modality or do not define any weights")

    def _check_weight(self, weight):
        len_w = len(self._weights)
        len_def = len([w for w in self._weights if w is not None])
        len_comp = (len_w - len_def) if weight is None else len_def
        if 0 < len_w < len_comp:
            raise ValueError("You should either define weights for each modality or do not define any weights")
        if weight is not None and weight <= 0:
            raise ValueError("Weight should be a positive number")

    def _append_emb_with_dist(self, emb, dist, weight, cols=None):
        self._check_weight(weight)
        self._weights[emb.name] = weight
        self._metrics[emb.name] = dist
        self._embeddings[emb.name] = emb
        self._used_columns |= cols if cols is not None else {emb.name}
        return self

    def vector(self,
               name: str,
               distance: Distance = Distance.EUCLIDEAN,
               weight: Optional[float] = None):
        """
        Append to pipeline vector modality handler

        :param name: name of a vector column
        :param distance: preferred inta modality distance metric, default :class`Distance.EUCLIDEAN`
        :param weight: modality importance.
        You should either specify importance for each modality,
        or let framework infer weights by specifying None for all
        """
        self._verify_name_usage(name)
        self._verify_type(name, (VectorType, OldVectorType))
        tab_emb = VectorColTabularEmb(name)
        return self._append_emb_with_dist(tab_emb, distance, weight)

    def tabular(self,
                name: str,
                features: Set[str],
                distance: Distance = Distance.EUCLIDEAN,
                weight: Optional[float] = None
                ):
        """
        Append to pipeline tabular modality handler, which will gather specified 'features' into single vector.

        :param name: new name for a column which will represent tabular modality
        :param features: column names to gather. Each column can contain either numeric or string values.
        If column has string type, it will be considered as categorical feature
        :param distance: preferred inta modality distance metric, default :class`Distance.EUCLIDEAN`
        :param weight: modality importance.
        You should either specify importance for each modality,
        or let framework infer weights by specifying None for all
        """
        self._verify_name_usage(name)
        for col_name in features:
            self._verify_not_used(col_name)
            self._verify_type(col_name, (NumericType, StringType))
        tab_emb = MultiColTabularEmb(name, features)
        return self._append_emb_with_dist(tab_emb, distance, weight, features)

    def _resolve_image(self, name, reduce_dim, model, root_path, batch_size) -> ImageEmb:
        if isinstance(model, TorchImages):
            return TorchImageEmb.from_zoo(name, reduce_dim, model, root_path, batch_size)
        if isinstance(model, TFImages):
            return TFImageEmb.from_zoo(name, reduce_dim, model, root_path, batch_size)
        raise ValueError(f"Failed to resolve image model for object {model}")

    def image(self,
              name: str,
              model,
              root_path: str,
              distance: Distance = Distance.EUCLIDEAN,
              reduce_dim: bool = True,
              batch_size: int = 32,
              weight: Optional[float] = None
              ):
        """
        Append to pipeline image modality handler, which will convert images into vector embeddings
        Specified column should contain paths to images. Paths should be relevant to 'root_path'

        :param name: column name in original dataset with images' paths
        :param model: model used for vector embeddings inference. See :class`TorchImages` and :class`TFImages`
        :param root_path: absolute path of root directory, which contains image files
        :param distance: preferred inta modality distance metric, default :class`Distance.EUCLIDEAN`
        :param reduce_dim: specifies whether output embeddings should run dimensionality reduction procedure
        :param batch_size: number of images to read and run single model inference, default 32
        :param weight: modality importance.
        You should either specify importance for each modality,
        or let framework infer weights by specifying None for all
        """
        self._verify_name_usage(name)
        image_emb = self._resolve_image(name, reduce_dim, model, root_path, batch_size)
        return self._append_emb_with_dist(image_emb, distance, weight)

    def _resolve_text(self, name, reduce_dim, model, batch_size) -> TextEmb:
        if isinstance(model, TorchTexts):
            return TorchTextEmb.from_zoo(name, reduce_dim, model, batch_size)
        if isinstance(model, TFTexts):
            return TFTextEmb.from_zoo(name, reduce_dim, model, batch_size)
        raise ValueError(f"Failed to resolve text model for object {model}")

    def text(self,
             name: str,
             model,
             distance: Distance = Distance.EUCLIDEAN,
             reduce_dim: bool = True,
             batch_size: int = 32,
             weight: Optional[float] = None
             ):
        """
        Append to pipeline textual modality handler, which will convert natural language into vector embeddings

        :param name: column name in raw dataset, that contains textual information (strings only)
        :param model: model used for vector embeddings inference. See :class`TorchTexts` and :class`TFTexts`
        :param reduce_dim: specifies whether output embeddings should run dimensionality reduction procedure
        :param distance: preferred inta modality distance metric, default :class`Distance.EUCLIDEAN`
        :param batch_size: number of rows for a single model inference, default 32
        :param weight: modality importance.
        You should either specify importance for each modality
        or let framework infer weights by specifying None for all
        """
        self._verify_name_usage(name)
        text_emb = self._resolve_text(name, reduce_dim, model, batch_size)
        return self._append_emb_with_dist(text_emb, distance, weight)

    def _make_pipeline(self):
        return [
            ClassColumn(self.class_col),
            FilterValues(self._used_columns),
            IdColumn(),
            Partition(self._partitions),
            *map(EmbWrapper, self._embeddings.values()),
            CountAmountAndDimensions(self._embeddings.keys()),
            ModalWeights(self._weights),
            JvmModalities(self._df._sc._jvm, self._metrics),
            Checkpoint(self.sc),
        ]

    def create(self) -> SparklingDF:
        """
        Executes pipeline and returns preprocessed :class`SparklingDF`
        """
        if len(self._embeddings) == 0:
            raise ValueError("No modalities were specified")
        t_start = SparklingLogger.start_preprocessing()

        monad = DataMonad('_df_preprocessor', self._make_pipeline())
        multimodal_df = monad.fit_transform(self._df)

        SparklingLogger.finish_preprocessing(t_start)

        return SparklingDF(
            df=multimodal_df,
            dist=monad['_jvm_dist'],
            modalities=monad['_modalities'],
            amount=monad['_amount']
        )
