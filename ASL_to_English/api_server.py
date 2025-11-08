from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import ASL_to_English.routes as routes
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)