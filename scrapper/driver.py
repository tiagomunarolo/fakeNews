from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

import time

SCROLL = "window.scrollTo(0, document.body.scrollHeight);"


class Driver(webdriver.Chrome):

    def __init__(self, url: str = None, headless=True):
        option = Options()
        option.headless = headless
        self.url = url

        super().__init__(
            executable_path=ChromeDriverManager().install(),
            options=option)

    def get_page(self, sleep_time=5):
        self.get(self.url)
        time.sleep(sleep_time)

    def scroll_page(self, sleep_time=5, scroll_pos=None):
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
        try:
            element = self.find_element(by=by, value=value)
            element.click()
            time.sleep(sleep_time)
            return True
        except NoSuchElementException:
            return False