import re
from fastapi import FastAPI, Request, File, UploadFile,Form
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
import torch
import requests
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptAvailable,TranscriptsDisabled
from gtts import gTTS
from transformers import T5ForConditionalGeneration, T5Tokenizer
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
    transcript_text = get_transcript(url)
    youtube_url = url
    output_path = "./output"
    
    if not transcript_text:
        download_audio(url, output_path="./output", filename="audio")
        audio_file_path = './output/audio.mp3'
        
        # API_KEY = os.getenv("API_KEY")
        # model_id = 'whisper-1'
        # language = "en"

        # with open(audio_file_path, 'rb') as audio_file:
        #     response = openai.Audio.translate(
        #         api_key=API_KEY,
        #         model=model_id,
        #         file=audio_file
        #     )
        translation_text = translate_text(audio_file_path, target_language='en')
    else:
        translation_text = transcript_text

    # Summarize the translated text
    summarized_transcript = summarize_transcript(translation_text, model_name="t5-base-multilingual-cased")
    summary_text = summarized_transcript
    print("summary_text :", summary_text)

    # summarizer = pipeline("summarization")
    # result = summarizer(translation_text, max_length=300, min_length=100, do_sample=False)
    # summary_text = result[0]['summary_text']
    # print("summary_text :", summary_text)

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

def translate_text(text, target_language='en'):
    if target_language == 'en':
        return text
    else:
        API_KEY = os.getenv("API_KEY")
        model_id = 'whisper-1'

        response = openai.Audio.translate(
            api_key=API_KEY,
            model=model_id,
            file=text  # Pass the audio file directly for translation
        )

        translation_text = response.text
        return translation_text
from transformers import T5ForConditionalGeneration, T5Tokenizer

def summarize_transcript(transcript, model_name="t5-small"):
    # Load pre-trained T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Define the transcript to be summarized
    input_text = "summarize: " + transcript

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary tokens into text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary_text

def get_transcript(url, target_language='en'):
    try:
        video_id = url.split("v=")[1]  # Extract video ID from the URL
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ""
        for line in transcript:
            text += line['text'] + " "
        
        # Translate transcript to English if necessary
        if target_language != 'en':
            translator = Translator()
            translation = translator.translate(text, dest='en')
            text = translation.text
        
        return text
    except NoTranscriptAvailable:
        return None
    except TranscriptsDisabled:
        return None

def translate_text(text, target_language='en'):
    if target_language == 'en':
        return text
    else:
        API_KEY = os.getenv("API_KEY")
        model_id = 'whisper-1'

        response = openai.Audio.translate(
            api_key=API_KEY,
            model=model_id,
            file=text  # Pass the audio file directly for translation
        )

        translation_text = response.text
        return translation_text

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