from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import ASL_to_English.signtalk as signtalk

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Train model
    print("Starting ASL Recognition API...")
    print("Training model from signtalk_data.pkl...")
    try:
        clf, id_to_label, hands = signtalk.train_model("ASL_to_English/signtalk_data.pkl")
        # Store in app state for access in routes
        app.state.classifier = clf
        app.state.id_to_label = id_to_label
        app.state.hands = hands
        print("API ready to accept requests!")
    except Exception as e:
        print(f"Error during startup: {e}")
        print("Server will start but /predict endpoint will not work until model is trained.")
    yield
    # Shutdown: Cleanup (if needed)
    pass

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import ASL_to_English.routes as routes
app.include_router(routes.router)