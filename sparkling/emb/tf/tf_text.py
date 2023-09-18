from enum import Enum


import pyspark.sql.functions as F
import pandas as pd
from pyspark.sql.types import ArrayType, FloatType

from sparkling.emb.text_emb import TextBatch, TextEmb

from sparkling.emb.tf.tf_wrapper import TFWrapper


class TFTexts(Enum):
    """ Some predefined nlp  models for preprocessing texts by tensorflow framework """

    ALBERT = ('albert-base-v2', 768, 63)                                       # 63Mb, dim 768 +
    BERT = ('bert-base-cased', 768, 527)                                       # 527Mb, dim 768
    BLENDERBOT_SMALL = ('facebook/blenderbot_small-90M', 512, 350)             # 350Mb, dim 512
    BYT5 = ('google/byt5-small', 512, 1229)                                    # 1.2Gb, dim 512
    CTRL = ('ctrl', 1280, 7987)                                                # 7.8Gb, dim 1280
    DEBERTA = ('microsoft/deberta-base', 768, 555)                             # 555Mb, dim 768
    DEBERTA_V2 = ('microsoft/deberta-v2-xlarge', 1536, 3625)                   # 3.54Gb, dim 1536
    DISTILBERT = ('distilbert-base-uncased', 768, 363)                         # 363Mb, dim 768
    DISTILGPT = ('distilgpt2', 768, 328)                                       # 328Mb, dim 768
    ELECTRA = ('google/electra-small-generator', 256, 70)                      # 70Mb, dim 256 +
    ELECTRA_D = ('google/electra-small-discriminator', 256, 54)                # 54Mb, dim 256 +
    ESM = ('facebook/esm2_t6_8M_UR50D', 320, 30)                               # 30Mb, dim 320 +
    FLAN_T5 = ('google/flan-t5-small', 769, 440)                               # 440Mb, dim TODO
    GPT = ('openai-gpt', 768, 466)                                             # 466Mb, dim 768
    LED = ('allenai/led-base-16384', 1024, 648)                                # 648Mb, dim 1024
    MBART = ('facebook/mbart-large-cc25', 1024, 2498)                          # 2.44Gb, dim 1024
    MOBILE = ('lordtt13/emo-mobilebert', 512, 86)                              # 86Mb, dim 512 +
    MPNET = ('microsoft/mpnet-base', 768, 536)                                 # 536Mb, dim 768
    REMBERT = ('google/rembert', 1152, 2355)                                   # 2.30Gb, dim 1152
    ROBERTA = ('roberta-base', 768, 657)                                       # 657Mb, dim 768
    ROBERTA_PRELAYERNORM = ('andreasmadsen/efficient_mlm_m0.40', 768, 1669)    # 1.63Gb, dim 768
    T5 = ('t5-small', 512, 242)                                                # 242Mb, dim 512
    T5_V1_1 = ('google/t5-v1_1-base', 512, 991)                                # 991Mb, dim TODO
    TRANSFORMER_XL = ('transfo-xl-wt103', 1024, 1321)                          # 1.29Gb, dim 1024
    XGLM = ('facebook/xglm-564M', 1024, 3400)                                  # 3.32Gb, dim 1024
    XLM = ('xlm-mlm-en-2048', 2048, 2990)                                      # 2.92Gb, dim 2048
    XLM_ROBERTA = ('cardiffnlp/twitter-roberta-base-emotion', 768, 501)        # 501Mb, dim 768
    XLNET = ('xlnet-base-cased', 1024, 565)                                    # 565Mb, dim 1024


class TFTextEmb(TextEmb):
    """ Wrapper for processing distributed text data by tensorflow nlp model """

    def __init__(self, name, model_name, orig_dim, reduce_dim, tf_text_model, tokenizer, batch_size):
        super().__init__(
            name,
            model_name=model_name,
            orig_dim=orig_dim,
            reduce_dim=reduce_dim,
            wrapper=TFWrapper(tf_text_model),
            tokenizer=tokenizer,
            batch_size=batch_size
        )

    def __str__(self):
        return f'TFText [{self.name}, model: {self.model_name}]'

    @staticmethod
    def from_zoo(name: str, reduce_dim: bool, model: TFTexts, batch_size: int) -> TextEmb:
        from transformers import TFAutoModel, AutoTokenizer

        checkpoint, orig_dim, _ = model.value
        tf_text_model = TFAutoModel.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return TFTextEmb(name, model.name, orig_dim, reduce_dim, tf_text_model, tokenizer, batch_size)

    @staticmethod
    def udf_builder(**kwargs):
        import tensorflow as tf

        wrapper = kwargs['wrapper']
        tokenizer = kwargs['tokenizer']
        batch_size = kwargs['batch_size']

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            token_shape = tf.shape(token_embeddings)
            expanded_mask = tf.expand_dims(attention_mask, -1)
            expanded_mask = tf.broadcast_to(expanded_mask, token_shape)
            expanded_mask = tf.cast(expanded_mask, tf.float32)
            sum_embeddings = tf.reduce_sum(token_embeddings * expanded_mask, 1)
            token_amounts = tf.reduce_sum(expanded_mask, 1)
            return sum_embeddings / token_amounts

        def udf_wrapper(texts: pd.Series) -> pd.Series:
            embeddings = list()
            for text_batch in TextBatch(texts, batch_size):
                inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors='tf')
                batch_embeddings = mean_pooling(wrapper.model(**inputs), inputs['attention_mask'])
                embeddings.extend(batch_embeddings.numpy().tolist())
            return pd.Series(embeddings)

        return F.pandas_udf(udf_wrapper, ArrayType(FloatType()))
