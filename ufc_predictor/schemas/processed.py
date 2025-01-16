from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ProcessedFighter(BaseModel):
    fighter_name: str
    first_name: str
    last_name: str
    nickname: Optional[str] = None
    height_inches: Optional[int] = None
    weight_pounds: Optional[int] = None
    reach_inches: Optional[int] = None
    stance: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    career_fights: int = Field(ge=0)  # Must be non-negative

class FightStats(BaseModel):
    knockdowns: int = Field(ge=0)
    significant_strikes_landed: int = Field(ge=0)
    significant_strike_accuracy: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    total_strikes_landed: int = Field(ge=0)
    takedowns_landed: int = Field(ge=0)
    takedown_accuracy: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    submission_attempts: int = Field(ge=0)
    reversals: int = Field(ge=0)
    control_time: str = "0:00"
    head_strikes_landed: int = Field(ge=0)
    body_strikes_landed: int = Field(ge=0)
    leg_strikes_landed: int = Field(ge=0)
    distance_strikes_landed: int = Field(ge=0)
    clinch_strikes_landed: int = Field(ge=0)
    ground_strikes_landed: int = Field(ge=0)

class ProcessedFight(BaseModel):
    event_name: str
    bout_order: int = Field(ge=1)
    fight_url: str
    fighter_name: str
    opponent_name: str
    fighter_stats: FightStats
    opponent_stats: FightStats
    result: Optional[str] = "Unknown"
    weight_class: Optional[str] = "Unknown"
    win_method: Optional[str] = "Unknown"
    ending_round: int = Field(ge=1, le=5, default=1)
    ending_time: Optional[str] = "0:00"
    referee_name: Optional[str] = "Unknown" 