import time
from typing import List
from selenium.webdriver.common.by import By
from crawler.driver import Driver
from src.logger.logging import get_logger
from bs4 import BeautifulSoup as Bs
import pandas as pd
from crawler import DATASET_PATH

SCROLL = "window.scrollTo(0, document.body.scrollHeight);"
HTML_PARSER = 'html.parser'
GET_MORE = "CARREGAR MAIS"
MAIN_PAGE = "https://veja.abril.com.br/coluna/sensacionalista"
logger = get_logger(__file__)


def build_dataset(data: List[dict]) -> None:
    """
    Builds Sensacionalista dataset given new articles
    :param data: List
    """
    df = pd.read_csv(f"{DATASET_PATH}/sensacionalista.csv", index_col=0)
    for article in data:
        df2 = pd.DataFrame(data=article)
        df = pd.concat([df, df2])
    df.to_csv(path_or_buf=f"{DATASET_PATH}/sensacionalista.csv")


def get_news(url: str = MAIN_PAGE) -> None:
    """
    Web scrap G1 page to get all news, given url
    :rtype: object
    """
    last_height = -1
    label = False

    # Opens driver context
    with Driver(url=url) as driver:
        # get base page
        driver.get_page()
        # while exists content to be extracted
        while True:
            # scroll to ge new content
            driver.scroll_page(scroll_pos="window.scrollY + 500", sleep_time=1)
            new_height = driver.get_height()
            if new_height == last_height:
                response = driver.find_element_and_click(
                    by=By.LINK_TEXT, value=GET_MORE)
                if not response:
                    response = [x for x in driver.find_elements(By.XPATH, "//button") if
                                x.accessible_name == 'CARREGAR MAIS']
                    if response:
                        response[0].click()
                        time.sleep(0.5)
                new_height = driver.get_height()
                if new_height == last_height or not response:
                    break
            last_height = new_height
            logger.info("Keep scrolling...")

        # extract all page links
        links = Bs(driver.page_source, features=HTML_PARSER).find_all('a', href=True)
        links = [link['href'] for link in links if link['href'].startswith(MAIN_PAGE)]
        links = list(set(links))
        data = []
        for _url in links:
            # for each link, open article, and get news
            driver.get(url=_url)
            text = Bs(driver.page_source, features=HTML_PARSER).find_all('article')
            summary = Bs(driver.page_source, features=HTML_PARSER).find_all('title')
            if not text or not summary:
                continue
            text = text[0].text.lower().replace("sensacionalista", "").replace("isento de verdade", ""),
            summary = summary[0].text
            data.append({"LABEL": [label], "RESUMO": [summary], "TEXTO": [text]})
        # builds final dataset
        build_dataset(data=data)


if __name__ == '__main__':
    get_news()
