import os
from elevenlabs import play
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

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