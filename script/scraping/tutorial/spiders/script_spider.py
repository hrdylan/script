import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor


class ScriptSpider(CrawlSpider):
	name = "www.imsdb.com"
	allow_domains = "www.imsdb.com"
	start_urls = ["https://www.imsdb.com/TV/Seinfeld.html"]
	rules = [
		Rule(LinkExtractor(allow=(r'/TV')),follow=True),
		Rule(LinkExtractor(allow=(r'/transcripts/Seinfeld')),callback="download_html"),
	]

	def download_html(self, response):
		"""Call back function to be used when link is found that contains a sienfeld script"""
		print("VISITED", response.url)

		# extract the name of the episode from the html
		name = response.url[34:-5]

		# write to local directory
		filename = "html_scripts/%s.html" % str(name)
		file = open(filename, "w")
		file.write(response.css("pre").extract_first())
		file.close()