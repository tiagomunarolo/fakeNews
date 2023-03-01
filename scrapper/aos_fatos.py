import time

import selenium.common.exceptions

from scrapper.driver import Driver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup, element
import pandas as pd
import re
import random

HTML_PARSER = 'html.parser'
TRUE_PAGE = "https://www.aosfatos.org/noticias/checamos/?page={}"
READ_MORE = {'class': ['entry-read-more-inline']}
NUM_PAGES = 163

CHECK_TYPES = ["FALSO", "DISTORCIDO", "VERDADEIRO", "CONTRADITÓRIO", "NÃO É BEM ASSIM"]


def break_conditions(e_, to_ignore, checks, index):
    if e_ in checks:
        return -1

    if isinstance(e_, element.Tag):
        if "Referências:" in e_.text:
            return -1
        elif e_.attrs == READ_MORE:
            to_ignore += re.sub(r"[\n\r\s]+", "", e_.text)
            if index == len(checks) - 1:
                return -1

    if isinstance(e_, element.NavigableString):
        if "Referências:" in e_.text or "LEIA TAMBÉM" in e_.text:
            return -1
        else:
            return 0

    return 1


def get_labels(soup):
    false = soup.find_all("img", attrs={"data-image-id": "falso.png"})
    true = soup.find_all("img", attrs={"data-image-id": "true.png"})
    dist = soup.find_all("img", attrs={"data-image-id": "distorcido.png"})
    impreciso = soup.find_all("img", attrs={"data-image-id": "impreciso.png"})
    checks = true + false + dist + impreciso

    if not checks:
        checks = soup.find_all("figcaption")
    if not checks:
        checks = soup.find_all("p", text=CHECK_TYPES)

    return checks, checks


def get_aos_fatos():
    df = pd.DataFrame(columns=['RESUMO', 'TEXTO', 'LABEL'])
    for page in range(1, NUM_PAGES):
        driver = Driver(url=TRUE_PAGE.format(page), headless=True)
        driver.get_page(sleep_time=3)
        news_list = BeautifulSoup(driver.page_source, HTML_PARSER). \
            find_all("a", attrs={"class": ['entry-item-card', 'entry-content']})
        driver.quit()
        to_ignore = ""
        for news_index, news in enumerate(news_list):
            url_ = "https://www.aosfatos.org" + news['href']
            driver = Driver(url=url_, headless=True)
            driver.get_page(sleep_time=2)
            e_ = driver.find_element(by=By.CLASS_NAME, value="default-container")
            html = e_.get_attribute('outerHTML')
            soup = BeautifulSoup(html, HTML_PARSER)
            summary = soup.find_all("h1")[0].text
            checks, labels = get_labels(soup)
            if not checks:
                print(f"Not found - {page} - {news_index + 1}")
                continue
            for index, e in enumerate(checks):
                label_ = "VERDADEIRO" if "VERDADEIRO" in str(e).upper() else "FALSO"
                for e_ in e.next_elements:
                    code = break_conditions(e_, to_ignore, checks, index)
                    if code == -1:
                        break
                    elif code == 0:
                        continue
                    text = e_.text.strip()
                    if len(text) > 50 and re.sub(r"[\n\r\s]+", "", e_.text) not in to_ignore:
                        data = {"RESUMO": [summary], "TEXTO": [text], "LABEL": [label_]}
                        df2 = pd.DataFrame(data=data)
                        df = pd.concat([df, df2])
            print(f"PAGE {page} - ({news_index + 1}/{len(news_list)})")
            df.to_csv("./aos_fatos.csv")
            driver.quit()


get_aos_fatos()
