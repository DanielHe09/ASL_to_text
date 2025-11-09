# Load environment variables FIRST, before any other imports
from dotenv import load_dotenv
import os

# Load .env file before importing anything that uses environment variables
load_dotenv('ASL_to_English/.env')

# Now import the app (which will import routes, which will import api_calls)
from ASL_to_English.api_server import app

#app.py doesn't run the server, just sets it up
#in server.py, we import the configured server from app.py, start the actual server using uvicorn, and make the server listen on port 8000

if __name__ == "__main__":
    import uvicorn

    #run the app on port 8000 (localhost)
    uvicorn.run(app, host="0.0.0.0", port=8000)