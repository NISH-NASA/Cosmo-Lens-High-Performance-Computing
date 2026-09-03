from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# For a hackathon, SQLite is perfect because it requires zero installation 
# and stores everything in a local file (cosmolens.db).
# You can easily swap this URL for a PostgreSQL database later if you deploy it!
SQLALCHEMY_DATABASE_URL = "sqlite:///./cosmolens.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
