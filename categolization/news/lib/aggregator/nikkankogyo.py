import urllib.request as req

from bs4 import BeautifulSoup


class NikkanKogyoScraper:

    def __init__(self):
        self.nikkan_url = 'https://www.nikkan.co.jp'
        html = req.urlopen(self.nikkan_url).read()
        self.soup = BeautifulSoup(html, 'lxml')

    def extract_category_title_and_content(self, url):
        res = req.urlopen(url)
        soup = BeautifulSoup(res, 'lxml')
        category = soup.select_one(
            'div.detail > p.category').string.strip(' []　')
        title = soup.select_one('div.detail > div.ttl').contents[0].strip()
        content_src = soup.select('div.article div.detail div.txt p')
        content = ''
        for p in content_src:
            if not p.get('class') == ['caption']:
                if not p.string == None:
                    content += p.contents[0].strip()

        return (category, title, content)

    def find_main_topnews_link(self):
        a = self.soup.select_one('div.topnews > div.ttl > a')
        main_top_link = a.attrs['href']
        return main_top_link

    def extract_links(self, link_ml, free=True, limited=False):
        links = []
        span = link_ml.select_one('span')
        if not span == None:
            article_limit = span.attrs['class'][0]
            if free == True and article_limit == 'limited_f':
                link = link_ml.select_one('a').attrs['href']
                links.append(link)
            if limited == True and article_limit == 'limited':
                link = link_ml.select_one('a').attrs['href']
                links.append(link)
        return links

    def find_topnews_links(self, free=True, limited=False):
        links = [self.find_main_topnews_link()]
        links_p = self.soup.select(
            'div.topnews_list > ul > li > div.txt > p.ttl')
        for p in links_p:
            links.extend(self.extract_links(p))
        return links

    def find_genre_news_links(self, free=True, limited=False):
        links = []
        links_div = self.soup.select('div.genrewrap > div.box')
        for div in links_div:
            title_p = div.select_one('p.ttl')
            if not title_p == None:
                link = self.extract_links(title_p, free, limited)
                links.extend(link)
            links_li = div.select('ul > li')
            if not links_li == None:
                for li in links_li:
                    links.extend(self.extract_links(li, free, limited))
        return links

    def scrape_headline_news(self, free=True, limited=False):
        topnews_links = self.find_topnews_links(free, limited)
        topnews = []
        for link in topnews_links:
            topnews_url = self.nikkan_url + link
            title_and_content = self.extract_category_title_and_content(
                topnews_url)
            topnews.append(title_and_content)
        return topnews

    def scrape_all_toppage_news(self, free=True, limited=False):
        news_links = self.find_genre_news_links()
        news = []
        for link in news_links:
            url = self.nikkan_url + link
            title_and_content = self.extract_category_title_and_content(url)
            news.append(title_and_content)
        return news


def main():
    nks = NikkanKogyoScraper()
    news = nks.scrape_all_toppage_news()
    for topic in news:
        print(topic[0], '\n', topic[1], '\n', topic[2], '\n\n')


if __name__ == '__main__':
    main()