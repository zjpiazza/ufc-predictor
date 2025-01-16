from pydantic import BaseModel
from typing import Optional

class RawEvent(BaseModel):
    event: str
    date: str
    location: str
    url: str

class RawFighterDetails(BaseModel):
    url: str
    first: str
    last: str
    nickname: Optional[str]

class RawFighterStats(BaseModel):
    url: str
    height: str
    weight: str
    reach: str
    stance: str
    dob: str

class RawFightDetails(BaseModel):
    event: str
    bout: str
    url: str

class RawFightStats(BaseModel):
    event: str
    bout: str
    fighter: str
    kd: str
    sig_str: str
    sig_str_pct: str
    total_str: str
    td: str
    td_pct: str
    sub_att: str
    rev: str
    ctrl: str
    head: str
    body: str
    leg: str
    distance: str
    clinch: str
    ground: str

class RawFightResult(BaseModel):
    event: str
    bout: str
    outcome: str
    method: str
    round: str
    time: str
    referee: str 