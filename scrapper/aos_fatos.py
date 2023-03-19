from scrapper.driver import Driver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup, element
from . import DATASET_PATH
import pandas as pd
import warnings

AOS_FATOS_PATH = f"{DATASET_PATH}/aos_fatos.csv"
HTML_PARSER = 'html.parser'
TRUE_PAGE = "https://www.aosfatos.org/noticias/checamos/?page={}"
READ_MORE = {'class': ['entry-read-more-inline']}
NUM_PAGES = 90

CHECK_TYPES = ["FALSO", "DISTORCIDO", "VERDADEIRO", "CONTRADITÓRIO", "NÃO É BEM ASSIM"]

warnings.filterwarnings(action="ignore", category=FutureWarning)


def break_conditions(e_, checks, index):
    """
    Break conditions to html parser
    :param e_: html elements
    :param checks: one of CHECK_TYPES
    :param index: index of element
    :return:
    """
    if e_ in checks:
        return -1

    if isinstance(e_, element.Tag):
        if "Referências:" in e_.text:
            return -1
        elif e_.attrs == READ_MORE:
            if index == len(checks) - 1:
                return -1

    if isinstance(e_, element.NavigableString):
        if "Referências:" in e_.text or "LEIA TAMBÉM" in e_.text:
            return -1
        else:
            return 0

    return 1


def get_labels(soup):
    """
    Search for image labels to get information
    :param soup:
    :return:
    """
    false = soup.find_all("img", attrs={"data-image-id": "falso.png"})
    true = soup.find_all("img", attrs={"data-image-id": "true.png"})
    dist = soup.find_all("img", attrs={"data-image-id": "distorcido.png"})
    impreciso = soup.find_all("img", attrs={"data-image-id": "impreciso.png"})
    nao_assim = soup.find_all("img", attrs={"data-image-id": "nao_e_bem_assim.png"})
    checks = true + false + dist + impreciso + nao_assim
    quotes_ = soup.find_all("blockquote", )
    return checks, quotes_


def get_aos_fatos():
    """
    Web scrapping aof fatos
    """
    df = pd.DataFrame(columns=['RESUMO', 'TEXTO', 'LABEL'])
    for page in range(1, NUM_PAGES):
        driver = Driver(url=TRUE_PAGE.format(page))
        driver.get_page(sleep_time=3)
        news_list = BeautifulSoup(driver.page_source, HTML_PARSER). \
            find_all("a", attrs={"class": ['entry-item-card', 'entry-content']})
        driver.quit()
        for news_index, news in enumerate(news_list):
            url_ = "https://www.aosfatos.org" + news['href']
            driver = Driver(url=url_)
            driver.get_page(sleep_time=2)
            e_ = driver.find_element(by=By.CLASS_NAME, value="default-container")
            html = e_.get_attribute('outerHTML')
            soup = BeautifulSoup(html, HTML_PARSER)
            summary = soup.find_all("h1")[0].text
            checks, quotes_ = get_labels(soup)
            if not checks or len(checks) != len(quotes_):
                print(f"Not found - {page} - {news_index + 1}")
                continue
            for index, e in enumerate(checks):
                label_ = True if "VERDADEIRO" in str(e).upper() else False
                for e_ in e.next_elements:
                    code = break_conditions(e_, checks, index)
                    if code == -1:
                        break
                    elif code == 0:
                        continue
                    label_update = label_
                    if str(e_) != str(quotes_[index]):
                        label_update = not label_
                    text = e_.text.strip()
                    if len(text) > 50:
                        data = {"RESUMO": [summary], "TEXTO": [text], "LABEL": [label_update]}
                        print(data)
                        df2 = pd.DataFrame(data=data)
                        df = pd.concat([df, df2])
            print(f"PAGE {page} - ({news_index + 1}/{len(news_list)})")
            df.to_csv(path_or_buf=AOS_FATOS_PATH, index_label=False)
            driver.quit()