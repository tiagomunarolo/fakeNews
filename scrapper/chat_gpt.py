import os
import openai
import random
import pandas as pd
import time
from model.logger import get_logger
from scrapper import DATASET_PATH

openai.api_key = os.getenv("OPENAI_API_KEY")

topics = ["vacinação", "horário eleitoral", "impostos", "educação",
          "saúde", "política", "crimes políticos", "prefeitura", "polícia",
          "armamento", "economia", "guerra", "relações internacionais",
          "democracia", "comunismo", "lei", "sistema eleitoral", "votação",
          "eleição", "constituição", "governo", "ministério",
          "direitos humanos", "religião", "corrupção", "crime", "liberdade de expressão",
          "ditadura", ]

logger = get_logger(__file__)


def generate_fake_data(num_news: int = 1000):
    """
    Generates fake data using ChatGPT
    """
    if os.path.exists(path=f"{DATASET_PATH}/chatgpt.csv"):
        df = pd.read_csv(f"{DATASET_PATH}/chatgpt.csv")
    else:
        df = pd.DataFrame(columns=['TEXT', 'LABEL'])

    for index in range(num_news):
        min_ = random.randint(50, 200)
        max_ = random.randint(200, 500)
        tema_index = index % len(topics)
        news = topics[tema_index]
        logger.info(f"Building fake_news {index + 1}: "
                    f"[{min_}/{max_} about {news}")
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": f"Construa uma noticia falsa com mais de {min_} "
                            f"palavras e menos de {max_} palavras sobre {news} no Brasil"}
            ]
        )

        text = completion.choices[0].message.content
        data = {"TEXT": [text], "LABEL": [False]}
        df2 = pd.DataFrame(data=data)
        df = pd.concat([df, df2])
        df = df[['TEXT', 'LABEL']]
        df.to_csv(path_or_buf=f"{DATASET_PATH}/chatgpt.csv", index_label=False)
        time.sleep(30)

    logger.info("FINALIZADO")
