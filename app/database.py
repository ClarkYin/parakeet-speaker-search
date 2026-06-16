from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    import os
    sql_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "init_db.sql")
    with open(sql_path) as f:
        ddl = f.read()
    statements = [s.strip() for s in ddl.split(";") if s.strip()]
    with engine.connect() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
        conn.commit()
