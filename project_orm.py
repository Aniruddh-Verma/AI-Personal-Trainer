import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Column,String,Integer,Float, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

base = declarative_base()

# we will create our users Table
class User(base):
    __tablename__ = 'users'
    id =Column(Integer, primary_key = True)
    email = Column(String(50), unique=True) 
    name = Column(String(50))
    password = Column(String(60))
    group = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow, nullable= False)

    def __repr__(self) -> str:
        return f"{self.id}|{self.name}|{self.group}"
if __name__ == "__main__":
    engine = create_engine('sqlite:///database.sqlite')
    base.metadata.create_all(engine)
