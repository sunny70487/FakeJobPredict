#在fastapi下配置＆使用數據庫
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

db_config = {
    'host' : 'mysql',
    'port' : '3306',
    'database' : 'practice',
    'username' : 'root',
    'password' : 'sunny70487'

}

#SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
SQLALCHEMY_DATABASE_URL = 'mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset=utf8'.format(**db_config)

engine = create_engine(
    #echo=True表示引擎將用repr()函數記錄所有語句及其參數列表到日誌
    SQLALCHEMY_DATABASE_URL,  encoding='utf8', echo=True
)
#在SQLAlchemy中，CRUD都是通過會話(session)進行的，所以須創建會話，每一個SessionLocal實例就是一個session
##flush()提交數據庫語句到數據庫，但數據庫不一定執行寫入磁盤內，commit()指提交事務，將變更保存到數據庫文件
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

#創建基本的映射類
Base = declarative_base(bind=engine,name='Base')


