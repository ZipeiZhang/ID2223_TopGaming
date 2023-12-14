# import gradio as gr
# from moviepy.editor import VideoFileClip
# from transformers import WhisperForConditionalGeneration, WhisperProcessor
# from transformers import pipeline
# import tempfile
# import os

# # Load your Whisper model
# model = WhisperForConditionalGeneration.from_pretrained("Zipei-KTH/whisper_CN")
# processor = WhisperProcessor.from_pretrained("Zipei-KTH/whisper_CN", language="chinese", task="transcribe")
# # processor = WhisperProcessor.from_pretrained("Zipei-KTH/whisper_CN")
# pipe = pipeline(model="Zipei-KTH/whisper_CN")

# def transcribe(audio_file=None, video_file=None):
#     # Determine if audio or video file is provided
#     file_path = audio_file if audio_file is not None else video_file

#     # Check if the file is a video
#     if file_path.endswith('.mp4'):
#         # Extract audio from video
#         with VideoFileClip(file_path) as video:
#             temp_dir = tempfile.mkdtemp()
#             temp_audio_path = os.path.join(temp_dir, 'temp_audio.wav')
#             video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')

#         # Transcribe the extracted audio
#         text = pipe(temp_audio_path)["text"]

#         # Clean up temporary files
#         os.remove(temp_audio_path)
#         os.rmdir(temp_dir)
#     else:
#         # If it's an audio file, transcribe directly
#         text = pipe(file_path)["text"]

#     return text

# # Define the Gradio interface
# iface = gr.Interface(
#     fn=transcribe, 
#     inputs=[
#         gr.Audio(type="filepath", label="Upload audio file"),
#         gr.Video(label="Upload .mp4 video file")
#     ],
#     outputs="text",
#     title="Whisper Small Chinese",
#     description="Realtime demo for Chinese speech recognition using a fine-tuned Whisper small model. Supports both audio and .mp4 video files."
# )

# iface.launch(share=True)


import gradio as gr
from moviepy.editor import VideoFileClip
from transformers import pipeline
import tempfile
import os
import requests
from pytube import YouTube
import gradio as gr
from moviepy.editor import VideoFileClip
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import pipeline
import tempfile
import os

import gradio as gr
from moviepy.editor import VideoFileClip
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import pipeline
import tempfile
import os
import yt_dlp as youtube_dl

# Load your Whisper model
model = WhisperForConditionalGeneration.from_pretrained("Zipei-KTH/whisper_CN_2")
processor = WhisperProcessor.from_pretrained("Zipei-KTH/whisper_CN_2", language="chinese", task="transcribe")
pipe = pipeline(model="Zipei-KTH/whisper_CN_2")

def download_video(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': tempfile.mktemp() + '.%(ext)s',
        'noplaylist': True,
        'verbose': True  #
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            video_file = ydl.prepare_filename(info)
            return video_file
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None


from moviepy.editor import AudioFileClip

def transcribe(audio_file=None, video_file=None, video_url=None, max_audio_length='60'):
    # Check if max_audio_length is not provided or empty, and set a default value
    if not max_audio_length:
        max_audio_length = '60'  # Default maximum length in seconds
    max_audio_length = float(max_audio_length)  # Convert to float

    if video_url:
        file_path = download_video(video_url)
    else:
        file_path = audio_file if audio_file is not None else video_file

    if file_path.endswith('.mp4'):
        with VideoFileClip(file_path) as video:
            # Truncate the video clip if it's longer than max_audio_length
            if video.duration > max_audio_length:
                video = video.subclip(0, max_audio_length)  # Keep only the first max_audio_length seconds

            temp_dir = tempfile.mkdtemp()
            temp_audio_path = os.path.join(temp_dir, 'temp_audio.wav')
            video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')

        text = pipe(temp_audio_path)["text"]

        os.remove(temp_audio_path)
        os.rmdir(temp_dir)
    else:
        text = pipe(file_path)["text"]

    if video_url:
        os.remove(file_path)

    return text


# Rest of your Gradio interface code


# Define the Gradio interface
iface = gr.Interface(
    fn=transcribe, 
    inputs=[
        gr.Audio(type="filepath", label="Upload audio file"),
        gr.Video(label="Upload .mp4 video file"),
        gr.Textbox(label="Or enter a video URL"),
        gr.Textbox(label="enter the maximum length")
    ],
    outputs="text",
    title="Whisper Small Chinese",
    description="Realtime demo for Chinese speech recognition using a fine-tuned Whisper small model. Supports audio, .mp4 video files, and video URLs."
)

iface.launch(share=True)
