from enum import Enum

import pandas as pd

import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType

from sparkling.emb.image_emb import ImageEmb, ImageBatch


class TorchImages(Enum):
    """ Some predefined cv models for preprocessing images by torch framework """

    BEIT = ("microsoft/beit-base-patch16-224-pt22k", 768, 368)                    # 368 Mb, dim 768
    BIT = ("google/bit-50", 2048, 120)                                            # 102 Mb, dim 2048
    CONVNEXT = ('facebook/convnext-tiny-224', 768, 115)                           # 115 Mb, dim 768
    CONVNEXT_V2 = ("facebook/convnextv2-tiny-1k-224", 768, 115)                   # 115 Mb, dim 768
    DEIT = ("facebook/deit-base-distilled-patch16-224", 768, 349)                 # 349 Mb, dim 768
    EFFICIENT_NET = ("google/efficientnet-b7", 2560, 267)                         # 267 Mb, dim 2560
    LEVIT = ("facebook/levit-128S", 384, 32)                                      # 32  Mb, dim 384
    MOBILENET_V1 = ("google/mobilenet_v1_1.0_224", 1024, 17)                      # 17  Mb, dim 1024
    MOBILENET_V2 = ("google/mobilenet_v2_1.0_224", 1280, 15)                      # 15  Mb, dim 1280
    MOBILE_VIT = ('apple/mobilevit-small', 640, 23)                               # 23  Mb, dim 640
    REGNET = ("facebook/regnet-y-040", 1088, 83)                                  # 83  Mb, dim 1088
    SWIN_TRANSFORMER = ("microsoft/swin-tiny-patch4-window7-224", 768, 113)       # 113 Mb, dim 768
    SWIN_TRANSFORMER_V2 = ("microsoft/swinv2-tiny-patch4-window8-256", 768, 113)  # 113 Mb, dim 768
    VAN = ("Visual-Attention-Network/van-base", 512, 107)                         # 107 Mb, dim 512
    VIT = ("google/vit-base-patch16-224-in21k", 768, 346)                         # 346 Mb, dim 768


class TorchImageEmb(ImageEmb):
    """ Wrapper for processing image distributed files by torch cv model """

    def __init__(self, name, model_name, orig_dim, reduce_dim, torch_image_model, processor, root_path, batch_size):
        super().__init__(
            name,
            model_name=model_name,
            orig_dim=orig_dim,
            reduce_dim=reduce_dim,
            model=torch_image_model,
            processor=processor,
            root_path=root_path,
            batch_size=batch_size
        )

    def __str__(self):
        return f'TorchImage [{self.name}, model: {self.model_name}]'

    @staticmethod
    def from_zoo(name: str, reduce_dim: bool, model: TorchImages, root_path: str, batch_size: int) -> ImageEmb:
        from transformers import AutoModel, AutoImageProcessor

        checkpoint, orig_dim, _ = model.value
        torch_model = AutoModel.from_pretrained(checkpoint)
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        return TorchImageEmb(name, model.name, orig_dim, reduce_dim, torch_model, processor, root_path, batch_size)

    @staticmethod
    def udf_builder(**kwargs):
        import torch

        model, processor = kwargs['model'], kwargs['processor']
        fs_builder, root_path = kwargs['fs_builder'], kwargs['root_path']
        batch_size = kwargs['batch_size']

        def udf_wrapper(paths: pd.Series) -> pd.Series:
            embeddings, fs = list(), fs_builder()
            images = ImageBatch(paths, fs, root_path, batch_size, processor, tensors='pt')
            with torch.no_grad():
                for inputs in images:
                    outputs = model(**inputs).pooler_output
                    batch_embeddings = outputs.numpy()
                    embeddings.extend(list(batch_embeddings))
            return pd.Series(embeddings)

        return F.pandas_udf(udf_wrapper, ArrayType(FloatType()))
