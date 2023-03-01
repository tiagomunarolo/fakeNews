from selenium.webdriver.common.by import By
from scrapper.driver import Driver
from bs4 import BeautifulSoup
import pandas as pd

SCROLL = "window.scrollTo(0, document.body.scrollHeight);"
HTML_PARSER = 'html.parser'
GET_MORE = "VEJA MAIS"
G1_PAGE = "https://g1.globo.com/fato-ou-fake/"


def get_g1_page():
    """
    Web scrap G1 page to get all news
    :rtype: object
    """
    driver = Driver(url=G1_PAGE)
    last_height = -1
    driver.get_page()

    try:
        while True:
            print("Keep scrolling...")
            driver.scroll_page()
            new_height = driver.get_height()
            if new_height == last_height:
                response = driver.find_element_and_click(
                    by=By.LINK_TEXT, value=GET_MORE)
                new_height = driver.get_height()
                if new_height == last_height or not response:
                    break
            last_height = new_height
    except Exception as e:
        print(e)

    element = driver.find_element(value="bstn-launcher")
    html = element.get_attribute('outerHTML')
    driver.quit()
    soup = BeautifulSoup(html, HTML_PARSER)
    df = pd.DataFrame(columns=['RESUMO', 'TEXTO'])

    for article in soup.find_all(class_='feed-post-body'):
        if not article.a:
            continue
        summary = article.find(class_='feed-post-body-resumo')
        if not summary:
            continue
        data = {"RESUMO": [article.a.text], "TEXTO": [summary.text]}
        df2 = pd.DataFrame(data=data)
        df = pd.concat([df, df2])

    df.to_csv(path_or_buf="./g1.csv")


get_g1_page()
