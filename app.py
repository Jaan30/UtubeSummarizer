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




 
@app.post("/submit_url",response_class=HTMLResponse)
async def submit_url(request: Request,url: str = Form(...), language: str = Form(...)): 
    # Process the URL and language data as needed
    print(f"Received URL: {url}, Language: {language}")
    
    youtube_url = url
    output_path = "./output"
    download_audio(youtube_url, output_path, filename="audio")
    
    
    # # URL of the file to transcribe
    # FILE_URL = "./output/audio.mp3"

    # transcriber = aai.Transcriber()
    # transcript = transcriber.transcribe(FILE_URL)



    # mixed_text = transcript.text
    # # Translate the mixed text to English
    # print("Original Text:", mixed_text)

    # translated_text = translate_text(mixed_text)
   
    API_KEY = os.getenv("API_KEY")
    model_id = 'whisper-1'
    language = "en"
    

    audio_file_path = './output/audio.mp3'
    audio_file = open(audio_file_path, 'rb')

    response = openai.Audio.translate(
        api_key=API_KEY,
        model=model_id,
        file=audio_file
    )
    translation_text = response.text
    # Print the results
    # print("Translated Text:", translation_text)
    
    summarizer = pipeline("summarization")

    result =summarizer(translation_text, max_length=250, min_length=100, do_sample=False)
    

    summary_text = result[0]['summary_text']
    print("summary_text :",summary_text)

    tts = gTTS(summary_text, lang='en')

    # Save the audio as a temporary file
    audio_file = "summary_audio.mp3"
    tts.save(audio_file)
    
    shutil.move(f"{audio_file}", "static/summary_audio.mp3") 
    
    context = {
        "request": request,
        "url":url,
        "summary_text":summary_text,
        "audio_file":audio_file
    }

    return templates.TemplateResponse("result.html", context)





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
    
    


























    # aai.settings.api_key = "098ba046cf1647ff89705531acd882bb"

    # transcriber = aai.Transcriber()

    # audio_url = "./output/audio.mp3"

    # transcript = transcriber.transcribe(audio_url)

    # prompt = "Provide a brief summary of the transcript."

    # result = transcript.lemur.task(prompt)

    # print(result.response)
    
