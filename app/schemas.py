from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    SK_ID_CURR: int
    EXT_SOURCE_1: Optional[float] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    DAYS_BIRTH: Optional[int] = None
    REG_CITY_NOT_LIVE_CITY: Optional[int] = None



from typing import Literal


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]