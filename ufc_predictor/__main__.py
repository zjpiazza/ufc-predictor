from ufc_predictor.data.scraping.scraper import UFCScraper
import anyio




scraper = UFCScraper()
anyio.run(scraper.run)