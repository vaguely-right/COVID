
import requests
import lxml.html as lh
import pandas as pd

#%% Modified after https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
url = r'https://newsinteractives.cbc.ca/coronavirustracker/'
#Create a handle, page, to handle the contents of the website
page = requests.get(url)
#Store the contents of the website under doc
doc = lh.fromstring(page.content)
#Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//tr')

#Check the length of the first 12 rows
[len(T) for T in tr_elements[:12]]

#%%
#Create empty list
col=[]
i=0
#For each row, store each first element (header) and an empty list
for t in tr_elements[0]:
    i+=1
    name=t.text_content()
    col.append((name,[]) )

#%% Modified after https://www.pluralsight.com/guides/extracting-data-html-beautifulsoup
from bs4 import BeautifulSoup
import requests

#%%
url = r'https://newsinteractives.cbc.ca/coronavirustracker/'

# Make a GET request to fetch the raw HTML content
html_content = requests.get(url).text

# Parse the html content
soup = BeautifulSoup(html_content, "lxml")
print(soup.prettify()) # print the parsed data of html

print(soup.title)
print(soup.title.text)

#%%
for link in soup.find_all("a"):
    print("Inner Text: {}".format(link.text))
    print("Title: {}".format(link.get("title")))
    print("href: {}".format(link.get("href")))

#%%
cases_table = soup.find("table", attrs={"class": "daily-table__tbody covid-data-table"})
cases_table_data = cases_table.tbody.find_all("tr")  # contains 2 rows

# Get all the headings of Lists
headings = []
for td in gdp_table_data[0].find_all("td"):
    # remove any newlines and extra spaces from left and right
    headings.append(td.b.text.replace('\n', ' ').strip())

print(headings)


#%% https://www.codementor.io/@dankhan/web-scrapping-using-python-and-beautifulsoup-o3hxadit4
import requests
from bs4 import BeautifulSoup

url = r'https://newsinteractives.cbc.ca/coronavirustracker/'

page = requests.get(url)

print(page.status_code)

print(page.content)

soup = BeautifulSoup(page.content)
print(soup.prettify())

#%% Trying my own
import requests
from lxml import html
from bs4 import BeautifulSoup
import pandas as pd

pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)

url = r'https://newsinteractives.cbc.ca/coronavirustracker/'
xp = '//*[@id="section-daily-new"]/div[2]'

page = requests.get(url)
doc = html.fromstring(page.content)
table = doc.xpath(xp)[0]

colnames = [i.text_content() for i in table.xpath('//th')]



soup = BeautifulSoup(page.text,'lxml')
table = soup.find('table',{})
table_data = table.tbody.find_all('tr')




/html/body/main/article/section[1]/section/section/section[3]/div[2]/table





