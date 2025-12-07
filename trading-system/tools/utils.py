from sqlalchemy import create_engine

# FUNCTION TO GET DB ENGINE
def get_db_engine(db_url: str):
    """
    Creates and returns a SQLAlchemy engine instance using the provided database URL.
    
    Args:
        db_url: The connection string.
    """
    print(f"Connecting to database...")
    return create_engine(db_url, pool_recycle=3600)
