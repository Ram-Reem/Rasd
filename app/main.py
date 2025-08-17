from fastapi import FastAPI
from app.routes import sentiment_route
from fastapi.middleware.cors import CORSMiddleware


# This is the main FastAPI application file
# It imports the sentiment analysis routes and includes them in the app
# The app title appears in the API docs (Swagger UI)
app = FastAPI(title="تحليل رضا العميل")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       
    allow_credentials=True,
    allow_methods=["*"],         
    allow_headers=["*"],         
)

app.include_router(sentiment_route.router)


