import tempfile
import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_silence
import librosa
from openai import OpenAI
import re
import speech_recognition as sr
from dotenv import load_dotenv
load_dotenv()

# Get the API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def generate_response(user_input):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "user",
        "content": """
                for the below text return integer total count of all filler words
                return only the total count 
          """ 
          +user_input
      },

    ],
    temperature=0,
    max_tokens=256,
    top_p=1
  )

  return response.choices[0].message.content


def compute_filler(audio_file):

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data)

    nWords = len(transcript.split())
    fillers = generate_response(transcript)

    pattern = r'\d+'
    
    matches = re.findall(pattern, fillers)
    
    nFillers =  int(matches[0])

    return nWords,nFillers


def compute_voice(audio):
    y, sr = librosa.load(audio)

    frame_length = 2048
    hop_length = 512
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    mean_energy = np.mean(energy)
    std_energy = np.std(energy)

    soft_threshold = mean_energy - 0.5 * std_energy
    medium_threshold = mean_energy + 0.5 * std_energy

    soft_duration = 0
    medium_duration = 0
    high_duration = 0

    for e in energy:
        if e < soft_threshold:
            soft_duration += hop_length / sr
        elif soft_threshold <= e < medium_threshold:
            medium_duration += hop_length / sr
        else:
            high_duration += hop_length / sr

    total_duration = librosa.get_duration(y=y, sr=sr)

    percentage_soft = (soft_duration / total_duration) * 100
    percentage_medium = (medium_duration / total_duration) * 100
    percentage_high = (high_duration / total_duration) * 100

    voice_duration = (soft_duration,medium_duration,high_duration)
    voice_percentage = (percentage_soft,percentage_medium,percentage_high)

    return voice_duration,voice_percentage

def compute_pauses(audio_file_path, min_silence_duration=1000, silence_threshold=-50):
    audio = AudioSegment.from_wav(audio_file_path)
    silent_chunks = detect_silence(audio, min_silence_len=min_silence_duration, silence_thresh=silence_threshold)
    silent_chunks_filtered = [chunk for chunk in silent_chunks if chunk[1] - chunk[0] >= min_silence_duration]
    
    num_pauses = len(silent_chunks_filtered)
    pause_durations = [((start / 1000), (stop / 1000)) for start, stop in silent_chunks_filtered]
    total_pause_time = sum((stop - start) / 1000 for start, stop in silent_chunks_filtered)
    
    return num_pauses, total_pause_time

def compute_audio(input_video_path, output_audio_path):
    video_clip = VideoFileClip(input_video_path)

    audio_clip = video_clip.audio

    audio_clip.write_audiofile(output_audio_path)

    audio_clip.close()
    video_clip.close()



def detect_eye_contact(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            return True
    
    return False

def video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames // fps
    cap.release()
    return duration_seconds

def split_video(video_path, output_folder, num_segments, fps):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_frames = total_frames // num_segments

    sum_eye = 0
    
    for i in range(num_segments):
        start_frame = i * segment_frames
        end_frame = start_frame + segment_frames
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        eye_contact_count = 0
        frame_count = 0
        eye_contact_percentage = 0
        
        while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(output_folder, f"segment_{i}_frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg")
            cv2.imwrite(frame_path, frame)
            
            if detect_eye_contact(frame):
                eye_contact_count += 1
            frame_count += 1
            
        eye_contact_p = (eye_contact_count / frame_count) * 100 if frame_count > 0 else 0

        sum_eye += eye_contact_p
        
    cap.release()

    eye_contact = sum_eye/num_segments

    return eye_contact

def compute_eye(video):
    output_folder = "output_frames"
    num_segments = 20
    fps = 30
    eye_contact_percentage = split_video(video, output_folder, num_segments, fps)

    duration = video_duration(video)

    return eye_contact_percentage,duration


def compute(video):
    print('started')
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video.read())
            file_path = temp_file.name
    audio = "audio.wav"
    print("started")
    eye_contact,duration_vid = compute_eye(file_path)
    compute_audio(file_path, audio)
    words,fillers = compute_filler(audio)
    num_pauses,total_pause_time = compute_pauses(audio)
    voice_duration,voice_Percentage = compute_voice(audio)
    soft,medium,high = voice_duration
    soft_p,medium_p,high_p = voice_Percentage
    #print('executing')
    wpm = int((words/duration_vid)*60)

    fpm = int((fillers/duration_vid)*60)

    avg_words,avg_fillers = 125,1

    if wpm > avg_words:
        words_feed = "GREAT"
    else:
        words_feed = "SLOW"

    if fpm > avg_fillers:
        fillers_feed = "HIGH"
    else:
        fillers_feed = "GREAT"

    avg_eye = 60
    if eye_contact > avg_eye:
        eye_feed = "GREAT"
    else:
        eye_feed = "LOW"

    avg_pause = 0.5

    if num_pauses > 0:
        pause_time = total_pause_time/num_pauses
    else:
        pause_time = 0

    if pause_time > avg_pause:
        pause_feed = "LONG"
    else:
        pause_feed = "GREAT"
    
    data = {
    "Duration": duration_vid,
    "Words": {
        "Total": words,
        "WPM": wpm,
        "Feed": words_feed,
        "Average": avg_words
    },
    "Filler Words": {
        "Total": fillers,
        "FPM": fpm,
        "Feed": fillers_feed,
        "Average": avg_fillers
    },
    "Eye Contact": {
        "Percentage": eye_contact,
        "Feed": eye_feed,
        "Average": avg_eye
    },
    "Pauses": num_pauses,
    "Pause Time": {
        "Total": total_pause_time,
        "Per Pause": pause_time,
        "Feed": pause_feed,
        "Average": avg_pause
    },
    "Voices": {
        "Soft": soft,
        "Soft Percentage": soft_p,
        "Medium": medium,
        "Medium Percentage": medium_p,
        "High": high,
        "High Percentage": high_p
    }
}
    # print(f"Duration: {duration_vid:.1f} s")
    # print(f"Words : {words} ({wpm} wpm {words_feed} Avg {avg_words} wpm )")
    # print(f"Filler Words : {fillers} ({fpm} fpm {fillers_feed} Avg {avg_fillers}/min)")
    # print(f"Eye Contact : {eye_contact:.2f}% ({eye_feed} Avg {avg_eye}%)")
    # print(f"Pauses : {num_pauses}")
    # print(f"Pause Time : {total_pause_time:.2f} s ({pause_time:.1f} s/Pause {pause_feed} Avg {avg_pause} s/Pause)")
    # print(f"Soft Voices : {soft:.2f} s, {soft_p:.2f}%")
    # print(f"Medium Voices : {medium:.2f} s, {medium_p:.2f}%")
    # print(f"High Voices : {high:.2f} s, {high_p:.2f}%")
    # print("* s - seconds , m - minutes ,wpm - words per minute , fpm - fillers per minute")
    #print(data)
    return data

