import os
import csv
import moviepy.video.io.VideoFileClip as mp
import speech_recognition as sr
from pydub import AudioSegment
import subprocess

# Set FFmpeg path
ffmpeg_path = '/opt/homebrew/bin/ffmpeg'
if not os.path.isfile(ffmpeg_path):
    raise EnvironmentError(f"FFmpeg not found at specified path: {ffmpeg_path}. Please check the path.")

os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_path

# Step 1: Extract audio from video
def extract_audio(video_file, output_audio_file):
    try:
        clip = mp.VideoFileClip(video_file)
        clip.audio.write_audiofile(output_audio_file)
    except Exception as e:
        print(f"Failed to extract audio from '{video_file}' using moviepy, attempting with ffmpeg: {e}")
        try:
            subprocess.run([ffmpeg_path, '-i', video_file, '-q:a', '0', '-map', 'a', output_audio_file], check=True)
        except Exception as ffmpeg_error:
            print(f"Failed to extract audio using ffmpeg, attempting to convert to MP4: {ffmpeg_error}")
            converted_file = video_file.rsplit('.', 1)[0] + '.mp4'
            try:
                subprocess.run([ffmpeg_path, '-i', video_file, converted_file], check=True)
                clip = mp.VideoFileClip(converted_file)
                clip.audio.write_audiofile(output_audio_file)
            except Exception as convert_error:
                raise RuntimeError(f"Failed to convert video to MP4 and extract audio: {convert_error}")

# Step 2: Split audio and recognize speech
def split_audio_and_recognize(input_audio_file, output_dir, metadata_file):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(input_audio_file)

    audio = audio.set_channels(1).set_frame_rate(16000)
    segment_length = 15 * 1000  # 15 seconds in milliseconds
    total_length = len(audio)

    # Append metadata header if the file doesn't exist
    if not os.path.isfile(metadata_file):
        with open(metadata_file, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter='|')
            writer.writerow(['filename', 'text transcription', 'start time (s)', 'end time (s)'])

    for start_time in range(0, total_length, segment_length):
        segment = audio[start_time:start_time + segment_length]
        segment_file = os.path.join(output_dir, f"segment_{start_time // 1000}.wav")
        segment.export(segment_file, format="wav")

        with sr.AudioFile(segment_file) as source:
            audio_data = recognizer.record(source)

            try:
                result = recognizer.recognize_google(audio_data, show_all=True)

                if isinstance(result, dict) and 'alternative' in result and len(result['alternative']) > 0:
                    alternatives = result['alternative'][0]['transcript']
                    words = alternatives.split()

                    current_time = start_time / 1000  # Convert to seconds
                    duration_per_word = segment.duration_seconds / len(words) if words else 0
                    word_found = False

                    for i, word in enumerate(words):
                        word_start_time = current_time
                        word_end_time = current_time + duration_per_word
                        word_segment = segment[int((i * duration_per_word) * 1000):int(((i + 1) * duration_per_word) * 1000)]
                        word_file = os.path.join(output_dir, f"{word}_{start_time // 1000}_{i}.wav")

                        if word.strip():  # Only export if word is not empty
                            word_segment.export(word_file, format="wav")
                            word_found = True
                            with open(metadata_file, mode='a', newline='') as csv_file:
                                writer = csv.writer(csv_file, delimiter='|')
                                writer.writerow([word_file, word, word_start_time, word_end_time])

                        current_time = word_end_time

                    if not word_found:
                        os.remove(segment_file)

            except sr.UnknownValueError:
                print(f"No speech detected in segment starting at {start_time // 1000}s")
                os.remove(segment_file)
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                os.remove(segment_file)

if __name__ == "__main__":
    video_file = input("Please enter the path to the video file: ").strip().replace("'", "")
    output_audio_file = "extracted_audio.wav"
    output_dir = "output_words"
    metadata_file = os.path.join(output_dir, "metadata.csv")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        extract_audio(video_file, output_audio_file)
    except FileNotFoundError:
        print(f"The video file '{video_file}' was not found. Please check the file path.")
        exit(1)
    except RuntimeError as e:
        print(f"An error occurred while extracting audio: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)

    split_audio_and_recognize(output_audio_file, output_dir, metadata_file)

    print("Dataset created successfully!")
