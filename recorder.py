import speech_recognition as sr
import os

def record_audio():
    r = sr.Recognizer()
    source = sr.Microphone()

    with source as mic:
        print("Recording...")
        audio = r.listen(mic)

    # Save the recorded audio to a file
    filename = "recorded_audio.wav"
    filepath = os.path.join(os.getcwd(), filename)
    with open(filepath, "wb") as f:
        f.write(audio.get_wav_data())

    print("Saved recorded audio to {}".format(filename))

if __name__ == '__main__':
    record_audio()