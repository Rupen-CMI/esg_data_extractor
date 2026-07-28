from pydantic import BaseModel

class KeyPlayer(BaseModel):
    market_name: str
    key_players: list[str]