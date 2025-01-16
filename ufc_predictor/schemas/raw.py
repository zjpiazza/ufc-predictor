from typing import Dict, List
from pydantic import BaseModel, Field

# Fight Details Models
class FighterStats(BaseModel):
    significant_strikes_landed: int = Field(ge=0)
    significant_strike_accuracy: float = Field(ge=0.0, le=1.0)
    takedowns_landed: int = Field(ge=0)
    takedown_accuracy: float = Field(ge=0.0, le=1.0)
    control_time: str

class FightDetails(BaseModel):
    url: str
    event: str
    bout: str
    weight_class: str
    method: str
    round: int = Field(ge=1, le=5)
    time: str
    referee: str
    winner: str
    fighter_stats: Dict[str, FighterStats]

class Event(BaseModel):
    event_id: str
    name: str
    date: str
    location: str
    url: str
    fight_urls: List[str] 

class RawFighterStats(BaseModel):
    url: str
    height: str
    weight: str
    reach: str
    stance: str
    dob: str
    slpm: float
    str_acc: float
    sapm: float
    str_def: float
    td_avg: float
    td_acc: float
    td_def: float
    sub_avg: float 