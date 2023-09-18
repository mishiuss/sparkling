from enum import Enum

import pandas as pd
import pyspark.sql.functions as F

from pyspark import SparkContext
from pyspark.sql.types import ArrayType, FloatType

from sparkling.emb.text_emb import TextEmb, TextBatch


class TorchTexts(Enum):
    """ Some predefined nlp models for preprocessing texts by torch framework """

    ALBERT = ('albert-base-v2', 768, 47)                                       # 47Mb, dim 768 +
    BART = ('valhalla/bart-large-sst2', 1024, 1670)                            # 1.63Gb, dim 1024
    BERT = ('bert-base-cased', 768, 436)                                       # 436Mb, dim 768
    BIGBIRD = ('l-yohai/bigbird-roberta-base-mnli', 768, 512)                  # 512Mb, dim 768
    BLENDERBOT_SMALL = ('facebook/blenderbot_small-90M', 512, 350)             # 350Mb, dim 512
    BLOOM = ('bigscience/bloom-560m', 64, 1147)                                # 1.12Gb, dim 64
    BYT5 = ('google/byt5-small', 512, 1228)                                    # 1.2Gb, dim 512
    CANINE = ('google/canine-c', 768, 529)                                     # 529Mb, dim 768
    CANINE_S = ('google/canine-s', 768, 529)                                   # 529Mb, dim 768
    CTRL = ('ctrl', 1280, 6707)                                                # 6.55Gb, dim 1280
    DEBERTA = ('microsoft/deberta-base', 768, 559)                             # 559Mb, dim 768
    DEBERTA_V2 = ('microsoft/deberta-v2-xlarge', 1536, 1823)                   # 1.78Gb, dim 1536
    DISTILBERT = ('distilbert-base-uncased', 768, 363)                         # 363Mb, dim 768
    DISTILGPT = ('distilgpt2', 768, 353)                                       # 353Mb, dim 768
    ELECTRA = ('google/electra-small-generator', 256, 54)                      # 54Mb, dim 256 +
    ELECTRA_D = ('google/electra-small-discriminator', 256, 54)                # 54Mb, dim 256 +
    ERNIE = ('nghuyong/ernie-2.0-base-en', 768, 438)                           # 438Mb, dim 768
    ESM = ('facebook/esm2_t6_8M_UR50D', 768, 31)                               # 31.4Mb, dim 768 +
    GPT = ('openai-gpt', 768, 479)                                             # 479Mb, dim 768
    GPT2 = ('microsoft/DialogRPT-updown', 768, 1556)                           # 1.52Gb, dim 768
    GPTNEO = ('EleutherAI/gpt-neo-1.3B', 2048, 5437)                           # 5.31Gb, dim 2048
    IBERT = ('kssteven/ibert-roberta-base', 768, 501)                          # 501Mb, dim 768
    LED = ('allenai/led-base-16384',  1024, 648)                               # 648Mb, dim 1024
    LONGFORMER = ('jpwahle/longformer-base-plagiarism-detection', 768, 595)    # 595Mb, dim 768
    MBART = ('facebook/mbart-large-cc25', 1024, 2499)                          # 2.44Gb, dim 1024
    MVP = ('RUCAIBox/mvp', 1024, 1669)                                         # 1.63Gb, dim 1024
    MOBILE = ('lordtt13/emo-mobilebert', 512, 85)                              # 85Mb, dim 512 +
    MPNET = ('microsoft/mpnet-base', 768, 532)                                 # 532Mb, dim 768
    NYSTROFORMER = ('uw-madison/nystromformer-512', 768, 437)                  # 437Mb, dim 768
    OPT = ('ArthurZ/opt-350m-dummy-sc', 768, 1352)                             # 1.32Gb, dim 768
    REMBERT = ('google/rembert', 1152, 2355)                                   # 2.30Gb, dim 1152
    ROCBERT = ('ArthurZ/dummy-rocbert-seq', 768, 469)                          # 469Mb, dim 768
    ROBERTA = ('roberta-base', 768, 501)                                       # 501Mb, dim 768
    ROBERTA_PRELAYERNORM = ('andreasmadsen/efficient_mlm_m0.40', 768, 1454)    # 1.42Gb, dim 768
    SBERT = ('sberbank-ai/sbert_large_nlu_ru', 768, 1751)                      # 1.71Gb, TODO
    SQUEEZEBERT = ('squeezebert/squeezebert-uncased', 768, 103)                # 103Mb, dim 768 +
    SWITCH_TRANSFORMER = ('google/switch-base-8', 768, 1270)                   # 1.24Gb, dim 768
    XMOD = ('facebook/xmod-base', 768, 3492)                                   # 3.41Gb, dim 768
    XGLM = ('facebook/xglm-564M', 1024, 1157)                                  # 1.13Gb, dim 1024
    XLM = ('xlm-mlm-en-2048', 2048, 2734)                                      # 2.67Gb, dim 2048
    XLM_RoBERTa = ('cardiffnlp/twitter-roberta-base-emotion', 768, 499)        # 499Mb, dim 768
    XLM_V = ('facebook/xlm-v-base', 768, 3195)                                 # 3.12Gb, TODO
    XLNET = ('xlnet-base-cased', 1024, 467)                                    # 467Mb, dim 1024
    YOSO = ('uw-madison/yoso-4096', 768, 510)                                  # 510Mb, dim 768


class TorchTextEmb(TextEmb):
    """ Wrapper for processing distributed text data by torch nlp model """

    def __init__(self, name, model_name, orig_dim, reduce_dim, text_model, tokenizer, batch_size):
        super().__init__(
            name=name,
            model_name=model_name,
            orig_dim=orig_dim,
            reduce_dim=reduce_dim,
            model=text_model,
            tokenizer=tokenizer,
            batch_size=batch_size
        )

    def __str__(self):
        return f'TorchText [{self.name}, model: {self.model_name}]'

    @staticmethod
    def from_zoo(name: str, reduce_dim: bool, model: TorchTexts, batch_size: int):
        from transformers import AutoModel, AutoTokenizer
        checkpoint, orig_dim, _ = model.value
        text_model = AutoModel.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return TorchTextEmb(name, model.name, orig_dim, reduce_dim, text_model, tokenizer, batch_size)

    @staticmethod
    def udf_builder(**kwargs):
        import torch
        model, tokenizer = kwargs['model'], kwargs['tokenizer']
        sc, batch_size = SparkContext.getOrCreate(), kwargs['batch_size']
        state_dict = sc.broadcast(model.state_dict())

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            return sum_embeddings / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        def udf_wrapper(texts: pd.Series) -> pd.Series:
            model.load_state_dict(state_dict.value)
            embeddings = list()
            with torch.no_grad():
                for text_batch in TextBatch(texts, batch_size):
                    inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors='pt')
                    batch_embeddings = mean_pooling(model(**inputs), inputs['attention_mask'])
                    embeddings.extend(batch_embeddings.numpy().tolist())
            return pd.Series(embeddings)

        return F.pandas_udf(udf_wrapper, ArrayType(FloatType()))

