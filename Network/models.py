from sqlalchemy import Column, String, Integer, func

from database import Base

class Data(Base):
    __tablename__ = 'testdata'
    index = Column(Integer, primary_key = True, index = True, autoincrement = True )
    title = Column(String, nullable = True)
    location = Column(String, nullable = True)
    department = Column(String, nullable = True)
    salary_range = Column(String, nullable = True)
