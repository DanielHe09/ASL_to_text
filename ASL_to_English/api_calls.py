import os
from io import BytesIO
from dotenv import load_dotenv
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai import types


# Load .env file (in case it wasn't loaded yet)
load_dotenv('ASL_to_English/.env')

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))  # GEMINI_API_KEY in env

#get audio from elevenlabs api
def get_audio(text_input):
    
    try:
        if(text_input == "isSay"):
            text_input = "is say"

        audio = client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_multilingual_v2",
            text=text_input,
            output_format="mp3_44100_128",

        )
        print("Audio generated successfully")
        return audio
    except Exception as e:
        print(f"Error getting audio: {e}")
        return None

#preferably .mp3 as input (accepts bytes)
def get_transcription(audio_bytes, mime):
    try:
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav"   # <--- WORKS in all versions
        )

        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Transcribe this audio. Use speaker labels if possible.",
                audio_part
            ]
        )

        return resp.text

    except Exception as e:
        print("Error transcribing audio:", e)
        return None
