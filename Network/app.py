#前端與後端對接端口
import uvicorn
from fastapi import FastAPI,Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from models import * 
from crud import get_data1
from load_data import main
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from model_class import Random_num

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])
# Base.metadata.create_all(bind = engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
@app.get('/')
def world():
    return f'Congratulation!! Welcome to web!!'

@app.post('/execute')
def execute(input : Random_num):
    #賦予class類別中的變數
    result = main(input.random_num)
    response = {'trueNum': int(result[1]), 'predictNum': int(result[0]), 'index': int(result[2]), 'predictResult': int(result[3])}
    return JSONResponse(content=response)

@app.post('/data')
def fakejob(limit : int = 10, db:Session = Depends(get_db)):
    data = get_data1(db = db, limit = limit)
    return data

@app.get('/test')
def fakejob(limit : int = 10, db:Session = Depends(get_db)):
    data = get_data1(db = db, limit = limit)
    return data



if __name__ == '__main__':
    uvicorn.run('app:app',host= '0.0.0.0', port=5000,reload=True)