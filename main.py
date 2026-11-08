import os
import struct
from flask import Flask, request, Response
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app) # Allows web-based frontends to call this API without security blocks

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    bits_per_sample, rate = 16, 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try: rate = int(param.split("=", 1)[1])
            except: pass
        elif param.startswith("audio/L"):
            try: bits_per_sample = int(param.split("L", 1)[1])
            except: pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    params = parse_audio_mime_type(mime_type)
    bits, rate, channels = params["bits_per_sample"], params["rate"], 1
    data_size = len(audio_data)
    bytes_per_sample = bits // 8
    block_align = channels * bytes_per_sample
    byte_rate = rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1,
        channels, rate, byte_rate, block_align, bits, b"data", data_size
    )
    return header + audio_data

@app.route("/", methods=["GET"])
def health_check():
    return "Camilabot API is active and listening on port 8080!"

@app.route("/talk", methods=["POST"])
def talk():
    # Accepts the text payload sent by your roleplay frontend
    data = request.get_json(force=True)
    transcript = data.get("text", "Hello, I am ready.")

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="voice_yxzcaxx6a1bs"
                )
            )
        ),
    )

    audio_data = bytearray()
    mime_type = ""
    
    for chunk in client.models.generate_content_stream(
        model="gemini-3.8-flash-lite-tts",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=transcript)])],
        config=config,
    ):
        if chunk.parts and chunk.parts[0].inline_data:
            inline = chunk.parts[0].inline_data
            audio_data.extend(inline.data)
            mime_type = inline.mime_type

    if not audio_data:
        return Response("Failed to generate audio", status=500)

    wav_bytes = convert_to_wav(bytes(audio_data), mime_type)
    return Response(wav_bytes, mimetype="audio/wav")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
