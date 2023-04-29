import os
import openai
import random
import pandas as pd
import time
from src.logger.logging import get_logger
from crawler import DATASET_PATH

openai.api_key = os.getenv("OPENAI_API_KEY")

topics = ["vacinação", "horário eleitoral", "impostos", "educação",
          "saúde", "política", "crimes políticos", "prefeitura", "polícia",
          "armamento", "economia", "guerra", "relações internacionais",
          "democracia", "comunismo", "lei", "sistema eleitoral", "votação",
          "eleição", "constituição", "governo", "ministério",
          "direitos humanos", "religião", "corrupção", "crime", "liberdade de expressão",
          "ditadura", ]

TRUE_CONTENT = "Faça um resumo de uma noticia real com mais de {min_} " \
               "e menos de {max_} palavras sobre {news} no Brasil cuja fonte seja BBC, UOL ou G1"

FAKE_CONTENT = "Construa uma noticia falsa com mais de {min_}" \
               " e menos de {max_} palavras sobre {news} no Brasil"

logger = get_logger(__file__)


def build_generic_data(num_news: int = 100000):
    """
    Generates fake data using ChatGPT
    """
    if os.path.exists(path=f"{DATASET_PATH}/chatgpt.csv"):
        df = pd.read_csv(f"{DATASET_PATH}/chatgpt.csv")
    else:
        df = pd.DataFrame(columns=['TEXT', 'LABEL'])

    for index in range(num_news):
        min_ = random.randint(50, 200)
        max_ = random.randint(201, 500)
        index_topic = index % len(topics)
        news_topic = topics[index_topic]
        # random fake/true news option
        tf = random.randint(0, 1)
        if tf == 0:
            true_fake, label_ = FAKE_CONTENT, False
        else:
            true_fake, label_ = TRUE_CONTENT, True

        logger.info(f"Building news({index + 1}): "
                    f"[{min_}/{max_}] about {news_topic} ==> {str(label_).upper()}")

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user",
                     "content": true_fake.format(min_=min_, max_=max_, news=news_topic)}
                ]
            )

            text = completion.choices[0].message.content
            data = {"TEXT": [text], "LABEL": [label_]}
            df2 = pd.DataFrame(data=data)
            df = pd.concat([df, df2])
            df = df[['TEXT', 'LABEL']]
            df.to_csv(path_or_buf=f"{DATASET_PATH}/chatgpt.csv", index_label=False)
            time.sleep(5)
        except openai.error.OpenAIError as e:
            logger.info(e)
            time.sleep(5)
            continue

    logger.info("DATASET_UPDATED")
