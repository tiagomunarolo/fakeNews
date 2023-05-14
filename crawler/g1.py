from typing import List, Union
from selenium.webdriver.common.by import By
from crawler.driver import Driver
from src.logger.logging import get_logger
from bs4 import BeautifulSoup as Bs
import pandas as pd
from crawler import DATASET_PATH

SCROLL = "window.scrollTo(0, document.body.scrollHeight);"
HTML_PARSER = 'html.parser'
GET_MORE = "VEJA MAIS"
G1_FATO_FAKE = "https://g1.globo.com/fato-ou-fake/"
G1_POLITICS = "https://g1.globo.com/politica/"
G1_SENSACIONALISTA = "https://oglobo.globo.com/blogs/humor/sensacionalista/"
logger = get_logger(__file__)


def build_dataset(data: List[dict], url: Union[bool, str] = False) -> None:
    """
    Builds G1 dataset given new articles
    :param data: List
    :param url: str, bool

    """
    file = "g1.csv"
    if url == G1_SENSACIONALISTA:
        file = "sensacionalista.csv"

    df = pd.read_csv(f"{DATASET_PATH}/{file}", index_col=0)
    for article in data:
        df2 = pd.DataFrame(data=article)
        df = pd.concat([df, df2])
    df.to_csv(path_or_buf=f"{DATASET_PATH}/{file}")


def get_news_g1(url: str = G1_POLITICS) -> None:
    """
    Web scrap G1 page to get all news, given url
    :rtype: object
    """
    last_height = -1
    if url == G1_SENSACIONALISTA:
        label = False
    else:
        label = True if url == G1_POLITICS else None

    links = set()
    # Opens driver context
    with Driver(url=url) as driver:
        # get base page
        driver.get_page()
        # while exists content to be extracted
        try:
            while True:
                # scroll to ge new content
                driver.scroll_page(sleep_time=1.5)
                links_page = Bs(driver.page_source, features=HTML_PARSER).find_all('a', href=True)
                links_page = [a['href'] for a in links_page if 'feed-post-link' in a.get('class', [])]
                for link in links_page:
                    links.add(link)
                new_height = driver.get_height()
                if new_height == last_height:
                    response = driver.find_element_and_click(
                        by=By.LINK_TEXT, value=GET_MORE)
                    new_height = driver.get_height()
                    if new_height == last_height or not response:
                        break
                last_height = new_height
                logger.info("Keep scrolling...")
        except:
            pass

        # extract all page links

        data = []
        for _url in list(links):
            # for each link, open article, and get news
            driver.get(url=_url)
            text = Bs(driver.page_source, features=HTML_PARSER).find_all('article')
            summary = Bs(driver.page_source, features=HTML_PARSER).find_all('title')
            if not text or not summary:
                continue
            text = text[0].text,
            summary = summary[0].text
            data.append({"LABEL": [label], "RESUMO": [summary], "TEXTO": [text]})
        # builds final dataset
        build_dataset(data=data, url=url)


if __name__ == '__main__':
    get_news_g1(G1_SENSACIONALISTA)
