import scrapy

class EventSpider(scrapy.Spider):
    name = 'event'
    start_urls = ['https://www.ufc.com/events']

    # Returns a list of events
    def parse(self, response):
        pass
