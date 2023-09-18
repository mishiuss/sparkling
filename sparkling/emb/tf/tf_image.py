from enum import Enum

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType

from sparkling.emb.image_emb import ImageEmb, ImageBatch
from sparkling.emb.tf.tf_wrapper import TFWrapper


class TFImages(Enum):
    """ Some predefined cv models for preprocessing images by tensorflow framework """

    CONVNEXT = ('facebook/convnext-tiny-224', 768, 115)                      # 115 Mb, dim 768
    DEIT = ('facebook/deit-base-distilled-patch16-224', 768, 350)            # 350 Mb, dim 768
    MOBILE_VIT = ('apple/mobilevit-small', 650, 23)                          # 23  Mb, dim 640
    REGNET = ('facebook/regnet-y-040', 1088, 84)                             # 84  Mb, dim 1088
    SWIN_TRANSFORMER = ('microsoft/swin-tiny-patch4-window7-224', 768, 113)  # 113 Mb, dim 768
    VIT = ('google/vit-base-patch16-224-in21k', 768, 346)                    # 346 Mb, dim 768


class TFImageEmb(ImageEmb):
    """ Wrapper for processing image distributed files by tensorflow cv model """

    def __init__(self, name, model_name, orig_dim, reduce_dim, tf_image_model, processor, root_path, batch_size):
        super().__init__(
            name,
            model_name=model_name,
            orig_dim=orig_dim,
            reduce_dim=reduce_dim,
            wrapper=TFWrapper(tf_image_model),
            processor=processor,
            root_path=root_path,
            batch_size=batch_size
        )

    def __str__(self):
        return f'TFImage [{self.name}, model: {self.model_name}]'

    @staticmethod
    def from_zoo(name: str, reduce_dim: bool, model: TFImages, root_path: str, batch_size: int) -> ImageEmb:
        from transformers import TFAutoModel, AutoImageProcessor

        checkpoint, orig_dim, _ = model.value
        tf_model = TFAutoModel.from_pretrained(checkpoint)
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        return TFImageEmb(name, model.name, orig_dim, reduce_dim, tf_model, processor, root_path, batch_size)

    @staticmethod
    def udf_builder(**kwargs):
        import tensorflow as tf

        wrapper, processor = kwargs['wrapper'], kwargs['processor']
        fs_builder, root_path = kwargs['fs_builder'], kwargs['root_path']
        batch_size = kwargs['batch_size']

        def udf_wrapper(paths: pd.Series) -> pd.Series:
            embeddings, fs = list(), fs_builder()
            images = ImageBatch(paths, fs, root_path, batch_size, processor, tensors='tf')
            for inputs in images:
                outputs = wrapper.model(**inputs).pooler_output
                batch_embeddings = outputs.numpy()
                embeddings.extend(list(batch_embeddings))
            return pd.Series(embeddings)

        return F.pandas_udf(udf_wrapper, ArrayType(FloatType()))
