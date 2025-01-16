import pytest
from ufc_predictor.data.scraping.fight_scraper import FightScraper

class TestFightScraper:
    @pytest.mark.asyncio
    async def test_get_fight_data(self):
        scraper = FightScraper()
        fight_data = await scraper.get_fight_data('http://ufcstats.com/fight-details/6f0f84075fa1a4b0')
        
        # Check required fields
        assert 'event_id' in fight_data
        assert 'fight_id' in fight_data
        assert 'fighter_id' in fight_data
        assert 'opponent_id' in fight_data
        assert 'date' in fight_data
        assert 'result' in fight_data
        assert 'method' in fight_data
        assert 'round' in fight_data
        assert 'time' in fight_data
        
        # Check fight stats fields
        assert 'sig_strikes_landed' in fight_data
        assert 'sig_strikes_attempted' in fight_data
        assert 'takedowns_landed' in fight_data
        assert 'takedowns_attempted' in fight_data
        assert 'sub_attempts' in fight_data
        assert 'rounds' in fight_data
        
        # Check actual values for this specific fight
        assert fight_data['fighter_id'] == 'd3df1add9d9a7efb'  # Derrick Lewis
        assert fight_data['opponent_id'] == 'd31a5546c0e9b213'  # Rodrigo Nascimento
        assert fight_data['method'] == 'KO/TKO'
        assert fight_data['round'] == 3
        assert fight_data['time'] == '0:49'
        assert fight_data['result'] == 'L'  # Lewis lost this fight
        
        # Check round data
        assert isinstance(fight_data['rounds'], list)
        # Should have data for 3 rounds since fight ended in round 3
        assert len(fight_data['rounds']) == 3
        
        # Check fight totals
        assert fight_data['sig_strikes_landed'] >= 0
        assert fight_data['sig_strikes_attempted'] >= fight_data['sig_strikes_landed']
        assert fight_data['takedowns_landed'] >= 0
        assert fight_data['takedowns_attempted'] >= fight_data['takedowns_landed']
        assert fight_data['sub_attempts'] >= 0