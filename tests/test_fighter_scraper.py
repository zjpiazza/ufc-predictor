import pytest
from ufc_predictor.data.scraping.fighter_scraper import FighterScraper

class TestFighterScraper:
    @pytest.mark.asyncio
    async def test_get_fighter_data(self):
        scraper = FighterScraper()
        fighter_data = await scraper.get_fighter_data('http://ufcstats.com/fighter-details/d3df1add9d9a7efb')
        
        # Check required fields
        assert 'fighter_id' in fighter_data
        assert 'name' in fighter_data
        assert 'height' in fighter_data
        assert 'weight' in fighter_data
        assert 'reach' in fighter_data
        assert 'stance' in fighter_data
        assert 'dob' in fighter_data
        assert 'last_updated' in fighter_data
        
        # Check data types
        assert isinstance(fighter_data['fighter_id'], str)
        assert isinstance(fighter_data['height'], (int, type(None)))
        assert isinstance(fighter_data['weight'], (int, type(None)))
        assert isinstance(fighter_data['reach'], (int, type(None))) 