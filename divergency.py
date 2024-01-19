from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from elevenlabs import generate, play, Voice, VoiceSettings, set_api_key
from pydub import AudioSegment
from pydub.playback import play
import threading
from ollama import Client


model = ChatOllama(model_name="wizard-vicuna-uncensored:30b")
speech = True
set_api_key("")


conv = [{"role": "system", "content": ""}]

def chat(input):
    conv.append({"role": "user", "content": input})
    # print(conversation)
    for part in Client().chat(model='wizard-vicuna-uncensored:30b', messages=conv, stream=True):
        try:
            print(part['message']['content'], end='', flush=True)
        except:
            print(".")

    
def main():    
    
    

    while True:       

        user_input = input("Enter a prompt: ")
        if not user_input:
            exit()
        print()
        try:
            response = chat(user_input)
            print(response)
        except Exception as e:
            print(e)
        


                    

if __name__ == '__main__':
    main()
    
