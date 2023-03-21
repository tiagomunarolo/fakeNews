from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

import time

SCROLL = "window.scrollTo(0, document.body.scrollHeight);"


class Driver(webdriver.Chrome):
    """
    Web crawler driver - Gets information via selenium
    """

    def __init__(self, url: str = None, headless=True):
        option = Options()
        option.headless = headless
        self.url = url

        super().__init__(
            executable_path=ChromeDriverManager().install(),
            options=option)

    def get_page(self, sleep_time=5):
        """
        Retrives hrml document form URL
        :param sleep_time:
        """
        self.get(self.url)
        time.sleep(sleep_time)

    def scroll_page(self, sleep_time=5, scroll_pos=None):
        """
        Executes selenium script to scroll page
        :param sleep_time: int
        :param scroll_pos: int scroll position
        """
        if scroll_pos:
            self.execute_script(f"window.scrollTo(0, {scroll_pos});")
        else:
            self.execute_script(SCROLL)
        time.sleep(sleep_time)

    def get_height(self):
        """
        Get current page height
        :rtype: object
        """
        element = self.find_element(by=By.TAG_NAME, value="body")
        return element.get_attribute("scrollHeight")

    def find_element_and_click(self, by, value, sleep_time=5):
        """
        find element in screen by value, and click on it
        :param by:
        :param value:
        :param sleep_time:
        :return:
        """
        try:
            element = self.find_element(by=by, value=value)
            element.click()
            time.sleep(sleep_time)
            return True
        except NoSuchElementException:
            return False
