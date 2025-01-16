from ufc_predictor.data.scraping.scraper import UFCScraper
from ufc_predictor.data.parsers.fighter_parser import FighterParser
from ufc_predictor.data.parsers.fight_parser import FightParser
from ufc_predictor.data.parsers.event_parser import EventParser
import time
# import asyncio
import anyio

scraper = UFCScraper()


start_time = time.time()
anyio.run(scraper.run)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# event_parser = EventParser('http://ufcstats.com/event-details/81ddc98fceb30086')
# event_data = asyncio.run(event_parser.get_event_data())
# print(event_data)