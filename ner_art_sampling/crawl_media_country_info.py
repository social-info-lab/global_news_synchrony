import time
from urllib import request
from bs4 import BeautifulSoup
import ssl
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep,ctime
from lxml import etree

ssl._create_default_https_context = ssl._create_unverified_context


# url="https://sources.mediacloud.org/#/collections/country-and-state"
# headers = headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) ' 'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36'}
#
# page = request.Request(url,headers=headers)
# page_info = request.urlopen(page).read().decode('utf-8')#打开Url,获取HttpResponse返回对象并读取其ResposneBody
#
# # 将获取到的内容转换成BeautifulSoup格式，并将html.parser作为解析器
# soup = BeautifulSoup(page_info, 'html.parser')
# # 以格式化的形式打印html
# print(soup.prettify())
#
# for x in soup.find_all('a',string = re.compile('National')):
#     print(x)

driver_path = r'/usr/local/bin/chromedriver'
driver = webdriver.Chrome(driver_path)
driver.get("https://sources.mediacloud.org/#/collections/country-and-state")
page_source = driver.page_source
# print(page_source)


username = driver.find_element(by=By.NAME, value="email")
password = driver.find_element(by=By.NAME, value="password")

username.send_keys("xchen4@umass.edu")

# password.send_keys("xxxx")
# password.send_keys(Keys.ENTER)

# login_button = driver.find_elements(by=By.CLASS_NAME, value="MuiButtonBase-root MuiButton-root MuiButton-contained app-button   MuiButton-containedPrimary")

# wait for 15 sec to input the password
time.sleep(15)


national_outlets = driver.find_elements(by=By.PARTIAL_LINK_TEXT, value="National")
nation_num = len(national_outlets)

for i in range(nation_num):
    national_outlets = driver.find_elements(by=By.PARTIAL_LINK_TEXT, value="National")

    cur_collection = national_outlets[i]
    url = cur_collection.get_attribute('href')
    driver.get(url)
    time.sleep(20)

    # download_button = driver.find_element(by=By.CLASS_NAME, value="actions")
    download_button = driver.find_element(by=By.XPATH, value="//body/div/div/div/div/div/div/div/div/div/div/div/div/div/button")
    download_button.click()
    driver.back()
    time.sleep(2)

    print(f"{i} nation has been finished...")



