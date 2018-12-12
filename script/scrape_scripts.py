import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class myspider(CrawlSpider):
	name = "www.imsdb.com"
	allow_domains = "www.imsdb.com"
	start_urls = ["https://www.imsdb.com/"]
	rules = (
    # Extract links matching 'category.php' (but not matching 'subsection.php')
    # and follow links from them (since no callback means follow=True by default).
    Rule(LinkExtractor(allow=('scripts', ), callback='parse_script'))
	)	

	def parse_script(self, response):
		print(response)