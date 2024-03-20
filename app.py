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
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import requests
import youtube_transcript_api
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
    
    download_audio(url, output_path="./output", filename="audio")
    audio_file_path = './output/audio.mp3'
    
    transcript_text = get_transcript(url, target_language='en')
    # refine_text = preprocess_transcript(transcript_text)
    youtube_url = url
    # output_path = "./output"
    
    
    if not transcript_text:
        
        translation_text = translate_audio( target_language='en')
    else:
        translation_text = transcript_text
        
    print(translation_text)


    
    def remove_special_characters(text):
        # Define the pattern to match any character that is not a letter, number, space, period, comma, exclamation mark, or question mark
        pattern = r'[^a-zA-Z0-9.,!?\- ]'
        
        # Replace all matches of the pattern with an empty string
        cleaned_text = re.sub(pattern, '', text)
        
        return cleaned_text

    translation_text= remove_special_characters(translation_text)

    def remove_specified_words_and_phrases(text, removal_list):
        # Create a regular expression pattern to match any of the specified words or phrases
        pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, removal_list)) + r')\b', flags=re.IGNORECASE)
        # Replace all matches of the pattern with an empty string
        cleaned_text = pattern.sub('', text)  
        return cleaned_text

    # Function to read removal list from file
    def read_removal_list_from_file(file_path):
        with open(file_path, "r") as file:
            removal_list = [line.strip() for line in file]
        return removal_list

    
    # Example usage
    removal_list = read_removal_list_from_file("removal_list.txt")
    translation_text = remove_specified_words_and_phrases(translation_text, removal_list)

    def remove_special_symbols_except_one(text):
        # Define the pattern to match any consecutive special symbols except one
        pattern = r'([^\w\s])\1+'
        
        # Replace all consecutive matches of the pattern with a single occurrence of the matched symbol
        cleaned_text = re.sub(pattern, r'\1', text)
        
        return cleaned_text

    translation_text = remove_special_symbols_except_one(translation_text)

    # Initialize the summarization pipeline
    summarizer = pipeline("summarization")

    # Define the maximum length of each chunk
    max_chunk_length = 1000  # Adjust as needed

    # Split the translation_text into smaller chunks
    chunks = [translation_text[i:i+max_chunk_length] for i in range(0, len(translation_text), max_chunk_length)]

    # Initialize an empty list to store the summaries
    summary_texts = []

    # Summarize each chunk separately
    for chunk in chunks:
        result = summarizer(chunk, max_length=20, min_length=10, do_sample=False)
        summary_texts.append(result[0]['summary_text'])

    # Concatenate the summaries to get the final summary for the entire text
    final_summary = " ".join(summary_texts)

    print("summary_text :",final_summary)

    
    # Generate audio for the summary
    tts = gTTS(final_summary, lang='en')

    # Save the audio as a temporary file
    audio_file = "summary_audio.mp3"
    tts.save(audio_file)

    # Move the audio file to the static directory
    shutil.move(f"{audio_file}", "static/summary_audio.mp3")

    #Embedded url formation
    def get_embedded_url(url):
        if "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        elif "watch?v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        else:
            raise ValueError("Invalid YouTube URL format")
        embedded_url = f"https://www.youtube-nocookie.com/embed/{video_id}"
        return embedded_url
        

    # Render the result.html template with the summary and audio file
    context = {
        "request": request,
        "url": get_embedded_url(url),
        "summary_text": final_summary,
        "audio_file": audio_file
    }

    return templates.TemplateResponse("result.html", context)

def translate_audio(target_language='en'):
    audio_file_path = './output/audio.mp3'
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
    audio_file_path = './output/audio.mp3'
    try:
        if "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        elif "watch?v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        else:
            raise ValueError("Invalid YouTube URL format")

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Extract the languages from the transcript list
        available_languages = [transcript.language for transcript in transcript_list]
        lang = get_language_code(available_languages[0].split('(')[0].strip())
   
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        text = " ".join(line['text'] for line in transcript)
        if lang!='en':
            text=translate_audio(target_language='en')
        return text

    except (NoTranscriptAvailable, TranscriptsDisabled):
        return None

def get_language_code(language_name):
    # You need to define language_map somewhere in your code
    return language_map.get(language_name)


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

# def preprocess_transcript(transcript):
    
#     # Remove punctuation
#     transcript = re.sub(r'[^\w\s]', '', transcript)
        
#     # Remove non-verbal expressions
#     non_verbal_expressions = ["[laughter]", "[music]", "[applause]"]
#     for expression in non_verbal_expressions:
#         transcript = transcript.replace(expression, "")
    
#     # Remove extra whitespace
#     transcript = ' '.join(transcript.split())
    
#     return transcript

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
