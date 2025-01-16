import bs4
from datetime import datetime
from aiohttp_client_cache import CachedSession
from typing import Optional
from .base_parser import BaseParser
from asyncer import asyncify
from tqdm.asyncio import tqdm

async def get_events_list(session):
    """Get list of UFC events."""
    url = "http://ufcstats.com/statistics/events/completed?page=all"
    
    # Need to await the response
    response = await session.get(url)
    # Get text content from response
    text = await response.text()
    
    soup = bs4.BeautifulSoup(text, 'html.parser')
    
    event_links = soup.find_all('a', class_='b-link b-link_style_black')
    return [link.get('href') for link in event_links]

class EventParser(BaseParser):

    async def get_event_data(self, pbar: tqdm) -> dict:
        await self.get_soup()
        
        event_name = await self.get_event_name()
        event_date = await self.get_event_date()
        event_location = await self.get_event_location()
        event_fight_ids = await self.get_event_fight_ids()


        pbar.update(1)
        return {
            'event_id': self.object_id,
            'event_name': event_name,
            'event_date': event_date,
            'event_location': event_location,
            'event_fight_ids': event_fight_ids,
        }
        

    async def get_event_name(self) -> str:
        event_name_element = self.soup.find('span', 'b-content__title-highlight')
        if not event_name_element:
            raise ValueError(f"Could not find event name for {self.url}")
        return event_name_element.get_text(strip=True)
    
    async def get_event_date(self):
        parent = self.soup.find('ul', class_='b-list__box-list')
        date_element = parent.find_all('li', class_='b-list__box-list-item')[0]
        date_text = date_element.get_text(strip=True)
        date_text = date_text.replace('Date:', '')
        return datetime.strptime(date_text.strip(), '%B %d, %Y')
    
    async def get_event_location(self):
        parent = self.soup.find('ul', class_='b-list__box-list')
        location_element = parent.find_all('li', class_='b-list__box-list-item')[1]
        location_text = location_element.get_text(strip=True)
        return location_text.replace('Location:', '').strip()
    
    async def get_event_fight_ids(self):
        fights_elements = self.soup.find_all('tr', class_='b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click')
        return [fight.get('data-link').split('/')[-1] for fight in fights_elements]
