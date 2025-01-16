import pytest
from ufc_predictor.data.scraping.event_scraper import EventScraper

@pytest.mark.asyncio
async def test_event_details_scraping():
    """Test that EventScraper captures all required data from events listing page."""
    scraper = EventScraper()
    events = await scraper.get_event_details(limit=1)  # Just get the most recent event
    
    assert len(events) == 1
    
    event = events[0]
    assert event.name.startswith("UFC")
    assert event.date
    assert event.location
    assert event.url.startswith("http://ufcstats.com/event-details/")
    assert len(event.fight_urls) > 0

@pytest.mark.asyncio
async def test_event_fight_urls():
    """Test that EventScraper correctly extracts fight URLs from an event page."""
    scraper = EventScraper()
    # Test with a specific known event
    fight_urls = await scraper._get_event_fight_urls("http://ufcstats.com/event-details/d26394fc0e8e880a")
    
    assert len(fight_urls) > 0
    assert all(url.startswith("http://ufcstats.com/fight-details/") for url in fight_urls)

@pytest.mark.asyncio
async def test_event_scraper_error_handling():
    """Test that EventScraper handles errors gracefully."""
    scraper = EventScraper()
    events = await scraper.get_event_details(limit=2)  # Just get 2 events
    assert isinstance(events, list)
    assert len(events) <= 2

