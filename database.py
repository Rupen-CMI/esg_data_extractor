from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import scoped_session, sessionmaker, DeclarativeBase
import os
from dotenv import load_dotenv

load_dotenv()

URL=os.getenv("DB_URL")

class Base(DeclarativeBase):
    pass

db = create_engine(
    URL,
    # Test each pooled connection with a lightweight ping before use, so a
    # connection the cloud DB closed while idle is transparently replaced
    # instead of blowing up the first request ("SSL connection closed").
    pool_pre_ping=True,
    # Proactively recycle connections older than 30 min (under typical server
    # idle timeouts) — belt-and-suspenders alongside pre_ping.
    pool_recycle=1800,
)
Sessionlocal = scoped_session(sessionmaker(autoflush=False, autocommit=False, bind=db))
try:
    with db.connect() as connection:
        result = connection.execute(text("SELECT 'connection successful!'"))
        print(result.fetchone()[0])
except Exception as e:
    print(f"Connection failed! {str(e)}")

def get_db():
    db = Sessionlocal()
    try:
        yield db
    except Exception as e:
        print(f"DB SESSION ERROR! {str(e)}")
        raise
    finally:
        db.close()
