from pydantic import BaseModel
from typing import Optional


class GroundTruth(BaseModel):
    uri: str
    width: int
    height: int
    res_change_factor: float
    is_vfr: bool
    pix_fmt: str
    framerate: float
    num_frames: int
    res_change_frame: Optional[int] = None
    broken_frame: Optional[int] = None
    timebase: float
    color_space: str
    color_range: str
