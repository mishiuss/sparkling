import json
import urllib

import numpy as np
import pandas as pd
import requests

from pyspark import Row
from pyspark.sql import SparkSession
from pyspark.sql.functions import split

from fetcher import MultiModalFetcher
from sparkling import SparklingDF, SparklingBuilder, Distance
from sparkling.emb.torch import TorchImages, TorchTexts


class Wiki(MultiModalFetcher):
    WIKI_CATEGORY = 'Category:21st-century American guitarists'
    WIKI_CSV_PATH = 'american_guitarists.csv'
    WIKI_REQUEST = 'http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles='

    IMAGES_DIR_PATH = '/user/multimodal/musicians/images'
    TEXTS_PATH = '/user/multimodal/musicians/data_v8.csv'

    IMAGE_MODEL = TorchImages.SWIN_TRANSFORMER
    TEXT_MODEL = TorchTexts.ALBERT

    def preprocessed_path(self) -> str:
        return '/user/multimodal/musicians/preprocessed'

    def modals_path(self) -> str:
        return 'examples/data/musicians/modals.json'

    def _extract_image(self, page):
        try:
            import wikipedia

            result = wikipedia.search(page.title, results=1)
            wikipedia.set_lang('en')
            wiki_page = wikipedia.WikipediaPage(title=result[0])

            response = requests.get(self.WIKI_REQUEST + wiki_page.title)
            json_data = json.loads(response.text)
            wiki_image = list(json_data['query']['pages'].values())[0]['original']['source']

            image_path = "musicians/images/" + "_".join(page.title.split()) + ".jpg"
            urllib.request.urlretrieve(wiki_image, image_path)
            return image_path
        except Exception:
            return np.nan

    def _collect_wiki(self):
        # To install needed modules run:
        import wikipedia  # pip install wikipedia
        import wikipediaapi  # pip install wikipedia-api

        # Collecting data from category:
        wiki_wiki = wikipediaapi.Wikipedia('en')
        cat, pages = wiki_wiki.page(self.WIKI_CATEGORY), []

        for s in list(cat.categorymembers.keys()):
            try:
                if wikipedia.page(s) in pages:
                    continue
                pages.append(wikipedia.page(s))
            except wikipedia.exceptions.PageError:
                print("Page not found: {}".format(s))
            except wikipedia.exceptions.DisambiguationError:
                print("Page name refer to many pages: {}".format(s))
            except TimeoutError:
                print("TimeoutError: {}".format(s))

        # Creating dataframe of information collected
        df = pd.DataFrame({"name": [], "image": [], "summary": []})
        for page in pages:
            if page.title in df["name"].values:
                continue
            row = {"name": page.title, "image": self._extract_image(page), "summary": page.summary}
            df = df.append(row, ignore_index=True)
        df.to_csv(self.WIKI_CSV_PATH)

    def from_raw(self, session: SparkSession) -> SparklingDF:
        # You should collect dataset first, use method _collect_wiki() above.
        # Do not forget to put dataframe and images into hdfs

        text_df = session.read.csv(self.TEXTS_PATH) \
            .withColumn('image', split('image', '/')[2])

        images_paths = session.sparkContext.wholeTextFiles(self.IMAGES_DIR_PATH) \
            .keys().map(lambda x: Row(img_path=x))

        image_df = session.createDataFrame(images_paths) \
            .withColumn('image', split('img_path', '/')[7])

        df = text_df.join(image_df, on=['image'])

        return SparklingBuilder(df, partitions='default') \
            .image('img_path', self.IMAGE_MODEL, '', Distance.EUCLIDEAN) \
            .text('summary', self.TEXT_MODEL, Distance.EUCLIDEAN) \
            .create()
