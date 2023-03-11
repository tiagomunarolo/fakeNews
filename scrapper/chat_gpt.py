import os
import openai
import random
import pandas as pd
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

temas = ["vacinação", "horário eleitoral", "impostos", "educação",
         "saúde", "política", "crimes políticos", "prefeitura", "polícia",
         "armamento", "economia", "guerra", "relações internacionais",
         "democracia", "comunismo", "lei", "sistema eleitoral", "votação",
         "eleição", "constituição", "governo", "ministério", "direitos humanos", "religião"]


def generate_fake_data():
    """
    Generates fake data using ChatGPT
    """
    if os.path.exists("./csv_data/chatgpt.csv"):
        df = pd.read_csv("./csv_data/chatgpt.csv")
    else:
        df = pd.DataFrame(columns=['TEXT', 'LABEL'])

    for index in range(1000):
        min_ = random.randint(50, 120)
        max_ = random.randint(121, 200)
        tema_index = random.randint(0, len(temas) - 1)
        news = temas[tema_index]

        print(f"Gerando notícia falsa {index + 1}: "
              f"[{min_}/{max_}] palavras sobre {news}")
        try:
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
            df.to_csv(path_or_buf="./csv_data/chatgpt.csv")
            time.sleep(30)
        except Exception as e:
            print(e)
            break
    print("FONALIZADO")


generate_fake_data()