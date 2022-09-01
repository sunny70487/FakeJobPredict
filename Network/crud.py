#對數據庫的操作
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import func, select
from sqlalchemy import desc
import models



def get_data1( db:Session, limit: int = 10):
    return db.query(models.Data).order_by(func.random()).limit(limit).all()