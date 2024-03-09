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
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import requests
import youtube_transcript_api
from transformers import BartForConditionalGeneration, BartTokenizer
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptAvailable, TranscriptsDisabled
from youtube_transcript_api._errors import NoTranscriptAvailable, TranscriptsDisabled
from gtts import gTTS
import shutil
from language_mappings import language_map

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
    transcript_text = get_transcript(url, target_language='en')
    youtube_url = url
    output_path = "./output"
    
    if not transcript_text:
        download_audio(url, output_path="./output", filename="audio")
        audio_file_path = './output/audio.mp3'
        translation_text = translate_audio(audio_file_path, target_language='en')
    else:
        translation_text = transcript_text
    
    min_length,max_length = get_min_max(translation_text)
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    # Tokenize the translation_text
    inputs = tokenizer([translation_text], max_length=1024, return_tensors="pt", truncation=True)

    # Generate summary using the BART model
    summary_ids = model.generate(inputs.input_ids, max_length=max_length, min_length=min_length, num_beams=4, early_stopping=True)

    # Decode the summary tokens back into text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print("Summary Text:", summary_text)
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
        text =response.text
    return text
    
def get_transcript(url, target_language='en'):
    try:
        video_id = url.split("v=")[1]
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Extract the languages from the transcript list
        available_languages = [transcript.language for transcript in transcript_list]
        lang = get_language_code(available_languages[0].split('(')[0].strip())

        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        text = " ".join(line['text'] for line in transcript)
        text = translate_text(text, target_language=target_language)
        return text

    except (NoTranscriptAvailable, TranscriptsDisabled):
        return None

def get_language_code(language_name):
    # You need to define language_map somewhere in your code
    return language_map.get(language_name)

def translate_text(text, target_language='en'):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text

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

def get_min_max(transcript):
    num_words = len(transcript)

    if num_words < 500:
        min_length, max_length = 100, 125
    elif num_words <= 1000:
        min_length, max_length = 125, 150
    elif num_words < 4000:
        min_length, max_length = 150, 200
    else:
        min_length, max_length = 175, 200

    return min_length, max_length

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
