#coding utf-8

from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep


# download chrome driver in
# https://chromedriver.storage.googleapis.com/index.html?path=2.33/
driver = webdriver.Chrome("./chromedriver")
main_url = "http://wiki.dbpedia.org/downloads-2016-10"
driver.get(main_url)
sleep(30)

html = driver.page_source
main_page = BeautifulSoup(html, 'lxml')

page_table = main_page.find('div', {'id': 'table_wrapper'})
rows = page_table.find('tbody').findAll('tr')

fwrite = open("./dbpedia_links.txt", 'w')
for row in rows:
    cols = row.findAll('td')
    category = cols[0].find('a').text
    fwrite.write(category + '\n')
    for url in cols[1].findAll('small'):
        url = url.find('a')
        if url.text == 'ttl':
            ttl_url = url['href']

    fwrite.write(ttl_url + '\n\n')

fwrite.close()
driver.close()
