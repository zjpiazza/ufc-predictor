import pandas as pd
from aiohttp_client_cache import CachedSession, SQLiteBackend


from ufc_predictor.data.parsers.event_parser import get_events_list, EventParser
from ufc_predictor.data.parsers.fight_parser import FightParser
from ufc_predictor.data.parsers.fighter_parser import FighterParser

from asyncer import asyncify
import asyncer
from pprint import pprint
from tqdm.asyncio import tqdm

class UFCScraper:
    def __init__(self):
        self.cache = SQLiteBackend(
            cache_name='ufc_cache',
            allowed_methods=('GET','POST'),
            expire_after=24 * 60 * 60  # Cache for 24 hours
        )
        self.session = None

    async def __aenter__(self):
        self.session = await CachedSession(cache=self.cache).__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.__aexit__(exc_type, exc_val, exc_tb)

    async def run(self):
        async with self:  # This will handle session creation and cleanup
            events = await self.scrape_events_data()
            
            event_fight_map = {}
            for event in events:
                event_fight_map[event['event_id']] = event['event_fight_ids']
                del event['event_fight_ids']
            
            fights = await self.scrape_fights_data(event_fight_map)


            per_round_fight_stats = []
            fight_details = []
            fighter_ids = []
            
            for fight in fights:
                per_round_fight_stats.extend(fight['rounds'])
                fight_details.append(fight['fight_details'])
                if fight['fight_details']['fighter1_id'] not in fighter_ids:
                    fighter_ids.append(fight['fight_details']['fighter1_id'])
                if fight['fight_details']['fighter2_id'] not in fighter_ids:
                    fighter_ids.append(fight['fight_details']['fighter2_id'])

            fighters = await self.scrape_fighters_data(fighter_ids)
            
            self.save_event_data(events)
            self.save_fights_data(fights)
            self.save_fighter_data(fighters)


    async def scrape_events_data(self, event_count: int = -1):
        events_list = await get_events_list(self.session)
        events = []
        
        # event_stop_index = event_count - len(events_list) if event_count != -1 else len(events_list)

        event_subset = events_list[:-30]
        pbar = tqdm(total=len(event_subset), desc="Scraping events")

        async with asyncer.create_task_group() as task_group:
            # Process each event
            for event in event_subset:  # Limiting to 3 events for testing
                # Get event data
                event_parser = EventParser(event, self.session, zyte=True)
                event_data = task_group.soonify(event_parser.get_event_data)(pbar=pbar)
                events.append(event_data)
        
        pbar.close()
        return [event.value for event in events]

    async def scrape_fights_data(self, event_fight_map: dict[str, list[str]]):
        fights = []

        total_fights = len([fight_id for fight_ids in event_fight_map.values() for fight_id in fight_ids])

        pbar = tqdm(total=total_fights, desc="Scraping fights")

        async with asyncer.create_task_group() as task_group:
            for event_id, fight_ids in event_fight_map.items():
                for fight_id in fight_ids:
                    fight_parser = FightParser(f"http://ufcstats.com/fight-details/{fight_id}", self.session, zyte=True)
                    fight_data = task_group.soonify(fight_parser.get_fight_data)(pbar=pbar, event_id=event_id)
                    fights.append(fight_data)

        pbar.close()
        return [fight.value for fight in fights]

    async def scrape_fighters_data(self, fighter_ids: list[dict]):
        fighters = []
        pbar = tqdm(total=len(fighter_ids), desc="Scraping fighters")
        
        async with asyncer.create_task_group() as task_group:
            for fighter_id in fighter_ids:
                fighter_url = f"http://ufcstats.com/fighter-details/{fighter_id}"
                fighter_parser = FighterParser(fighter_url, self.session, zyte=True)
                fighter_data = task_group.soonify(fighter_parser.get_fighter_data)(pbar=pbar)
                fighters.append(fighter_data)
        
        pbar.close()
        return [fighter.value for fighter in fighters]

    def save_event_data(self, events_data: list[dict]):
        events_df = pd.DataFrame(events_data)
        events_df.to_csv('data/raw/events.csv', index=False)

    def save_fights_data(self, fights: list[dict]):
        fight_details = []
        fight_stats_by_round = []

        for fight in fights:

            hydrated_fight_details = fight['fight_details'].copy()
            hydrated_fight_details['event_id'] = fight['event_id']
            fight_details.append(hydrated_fight_details)

            for round in fight['rounds']:
                round['event_id'] = fight['event_id']
                fight_stats_by_round.append(round)
        
        fight_details_df = pd.DataFrame(fight_details)
        fight_stats_by_round_df = pd.DataFrame(fight_stats_by_round)
        fight_details_df.to_csv('data/raw/fights.csv', index=False)
        fight_stats_by_round_df.to_csv('data/raw/fight_stats_by_round.csv', index=False)
    
    def save_fighter_data(self, fighters: list[dict]):
        fighters_df = pd.DataFrame(fighters)
        fighters_df.to_csv('data/raw/fighters.csv', index=False)


    def save_fights_data_old(self, events_data: list, all_fight_data: dict):
        fight_rows = []
        
        for event in events_data:
            event_id = event['event_url'].split('/')[-1]
            
            # Get total number of fights to calculate correct fight order
            total_fights = len(event['event_fight_ids'])
            
            for fight_id in event['event_fight_ids']:
                fight_data = all_fight_data.get(fight_id, {})
                fight_details = fight_data.get('fight_details', {})
                
                # Calculate fight order (reverse of index)
                fight_order = total_fights - event['event_fight_ids'].index(fight_id)
                
                fight_row = {
                    'fight_id': fight_id,
                    'event_id': event_id,
                    'fighter1_id': fight_data.get('fighter1_id'),
                    'fighter2_id': fight_data.get('fighter2_id'),
                    'fighter1_name': fight_data.get('fighter1_name'),
                    'fighter2_name': fight_data.get('fighter2_name'),
                    'weight_class': fight_details.get('weight_class'),
                    'gender': fight_details.get('gender'),
                    'method': fight_details.get('method'),
                    'final_round': fight_details.get('round'),
                    'final_time': fight_details.get('time'),
                    'format': fight_details.get('format'),
                    'referee': fight_details.get('referee'),
                    'fight_order': fight_order
                }
                fight_rows.append(fight_row)
        
        fights_df = pd.DataFrame(fight_rows)
        fights_df.to_csv('data/raw/fights.csv', index=False)

    def save_fight_stats(self, fight_stats: list):
        df = pd.DataFrame(fight_stats)
        df.to_csv('data/raw/fight_stats.csv', index=False)

    def process_fight_stats(self, fight_id: str, fight_data: dict) -> list:
        stats_rows = []
        
        # Get fighter names from the first round
        fighters = list(fight_data['rounds'][1].keys())
        fight_details = fight_data['fight_details']
        
        # Process each round's stats
        for round_num in fight_data['rounds']:
            round_stats = fight_data['rounds'][round_num]
            
            # Process each fighter's stats
            for fighter in fighters:
                fighter_stats = {
                    'fight_id': fight_id,
                    'fighter_id': round_stats[fighter]['fighter_id'],
                    'round': round_num,
                    'weight_class': fight_details['weight_class'],
                    'method': fight_details['method'],
                    'final_round': fight_details['round'],
                    'final_time': fight_details['time'],
                    'format': fight_details['format'],
                    'referee': fight_details['referee'],
                }
                fighter_stats.update(round_stats[fighter])
                stats_rows.append(fighter_stats)
        
        return stats_rows

def main():
    scraper = UFCScraper()
    scraper.scrape()

if __name__ == "__main__":
    main()


    