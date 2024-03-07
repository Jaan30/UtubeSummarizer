import re
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv
from pytube import YouTube
import uvicorn
import fastapi
import assemblyai as aai
from googletrans import Translator
from pydantic import BaseModel
import openai
from transformers import pipeline
from summarizer import Summarizer
from langdetect import detect
import torch
import requests
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptAvailable, TranscriptsDisabled
from gtts import gTTS
import shutil

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class URLItem(BaseModel):
    url: str
    language: str

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/result')
def index(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})

@app.post("/submit_url", response_class=HTMLResponse)
async def submit_url(request: Request, url: str = Form(...), language: str = Form(...)): 
    # Process the URL and language data as needed
    print(f"Received URL: {url}, Language: {language}")
    transcript_text = get_transcript(url,target_language='en')
    youtube_url = url
    output_path = "./output"
    
    if not transcript_text:
        download_audio(url, output_path="./output", filename="audio")
        audio_file_path = './output/audio.mp3'
        translation_text = translate_audio(audio_file_path, target_language='en')
    else:
        translation_text = transcript_text

    summarizer = pipeline("summarization") 
    result = summarizer(translation_text, max_length=250, min_length=100, do_sample=False)
    summary_text = result[0]['summary_text']
    print("summary_text :", summary_text)
    
    # Generate audio for the summary
    tts = gTTS(summary_text, lang='en')

    # Save the audio as a temporary file
    audio_file = "summary_audio.mp3"
    tts.save(audio_file)

    # Move the audio file to the static directory
    shutil.move(f"{audio_file}", "static/summary_audio.mp3")

    # Render the result.html template with the summary and audio file
    context = {
        "request": request,
        "url": url,
        "summary_text": summary_text,
        "audio_file": audio_file
    }

    return templates.TemplateResponse("result.html", context)

def translate_audio(audio_file_path, target_language='en'):
    API_KEY = os.getenv("API_KEY")
    model_id = 'whisper-1'

    with open(audio_file_path, 'rb') as audio_file:
        response = openai.Audio.translate(
            api_key=API_KEY,
            model=model_id,
            file=audio_file,
            target_language=target_language
        )

    return response.text

def get_transcript(url, target_language='en'):
    try:
        video_id = url.split("v=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Check if any transcript is available
        if not transcript:
            return None

        text = ""
        for line in transcript:
            text += line['text'] + " "

        # Check if English transcript exists or matches target language
        if 'en' in [t['language'] for t in transcript] and target_language == 'en':
            return text.strip()  # Return English transcript directly

        # If no English transcript or a different target language is needed, translate
        translator = Translator()
        detected_language = translator.detect(text).lang  # Detect the actual language
        translation = translator.translate(text, src=detected_language, dest=target_language)
        return translation.text

    except NoTranscriptAvailable:
        print("No transcript available for the video.")
        return None

    except TranscriptsDisabled:
        print("Transcripts are disabled for the video.")
        return None
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# def get_transcript(url):
#     try:
#         video_id = url.split("v=")[1]  # Extract video ID from the URL
#         transcript = YouTubeTranscriptApi.get_transcript(video_id)
#         text = ""
#         for line in transcript:
#             text += line['text'] + " "

#         # Detect the language of the transcript
#         transcript_language = detect(text)
    
#         if transcript_language != 'en':
#             translator = Translator()
#             translation = translator.translate(text, src=transcript_language, dest='en')
#             text = translation.text
#             return text
#         else:
#             return text
#     except NoTranscriptAvailable:
#         return None
#     except TranscriptsDisabled:
#         return None

def download_audio(youtube_url, output_path, filename="audio"):
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(output_path)

    # Get the default filename
    default_filename = audio_stream.default_filename

    # Rename the downloaded file
    downloaded_file_path = os.path.join(output_path, default_filename)
    new_file_path = os.path.join(output_path, f"{filename}.mp3")
    os.rename(downloaded_file_path, new_file_path)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
