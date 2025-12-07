import datetime
from datetime import datetime, timezone
def align_to_candle_time(dt: datetime, interval_hours: int = 1) -> datetime:
    """
    Aligns a datetime object to the start of the nearest hourly candle.
    
    Args:
        dt: The datetime object to align.
        interval_hours: The interval size in hours (e.g., 1 for hourly, 4 for 4-hour candles).
        
    Returns:
        The aligned datetime object (e.g., 2023-10-28 10:00:00).
    """
    aligned = dt.dt.floor('H')
    if interval_hours > 1:
        hour = aligned.hour
        aligned = aligned.replace(hour=(hour // interval_hours) * interval_hours)
    return aligned