from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader, JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from elevenlabs import generate, play, Voice, VoiceSettings, set_api_key
from pydub import AudioSegment
from pydub.playback import play
import threading
from ollama import Client
import chromadb
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pynput import keyboard

speech = True
set_api_key("")
conversation = [{"role": "system", "content": "Using context provided, Answer all questions asked of you as outlined."}]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.half

model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.half, low_cpu_mem_usage=True, use_safetensors=True
)
model.to("cpu")
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

def speech_to_text(key):
    if key == keyboard.Key.down:

        sample = dataset[0]["audio"]

        result = pipe("recorded_audio.wav")
        print(result["text"])

    

def chat(context, input):
    inquiry =f"Using the following pieces of context, answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible.\n\n{context}\n\nQuestion:{input}"
    conversation.append({"role": "user", "content": inquiry})
    # print(conversation)
    for part in Client().chat(model='llama2', messages=conversation, stream=True):
        try:
            print(part['message']['content'], end='', flush=True)
        except:
            print(".")

def main():

        
    
    #Load directory of documents for LUNA context - multi-document
    #loader = DirectoryLoader("/home/equious/Documents/AI-Testing/AI-Scripts", glob="**/*.md", use_multithreading=True)
    #Loads document for LUNA context - single markdown
    #loader = UnstructuredMarkdownLoader("3-rekt-test.md")
    loader = JSONLoader("test_data.json", text_content=False, jq_schema='{"$schema": "http://json-schema.org/draft-07/schema","type": "object","properties": {"id": {"type": "number"},"issue_protocol": {"type": "object","properties": {"id": {"type": "number"},"name": {"type": "string"},"category_list": {"type": "array","items": {"type": "object"}}}},"title": {"type": "string"},"content": {"type": "string"},"kind": {"type": "string"},"issue_source": {"type": "object","properties": {"name": {"type": "string"},"url": {"type": "string"},"logo_square_url": {"type": "string"},"logo_horizontal_url": {"type": "string"},"has_contest": {"type": "boolean"}}},"impact": {"type": "string"},"tag_list": {"type": "array","items": {"type": "object"}},"user_note": {"type": "object","properties": {"created_at": {"type": "string"},"note": {"type": "string"}}},"slug": {"type": "string"}},"required": ["id", "issue_protocol", "title", "content", "kind", "issue_source", "impact", "tag_list", "user_note", "slug"]}')
    # files = os.listdir("/home/equious/Documents/AI-Testing/AI-Scripts")
    # loaders = [UnstructuredMarkdownLoader(file) for file in files]

    data = loader.load()
    

    vectorstore = Chroma.from_documents(documents=data, embedding=OllamaEmbeddings())
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    
    listener = keyboard.Listener(on_release=speech_to_text)

    while True:       

        user_input = input("Enter a prompt: ")
        if not user_input:
            exit()
        print()
        try:
            retrieved_docs = retriever.invoke(user_input)
            response = chat(retrieved_docs, user_input)
            print(retrieved_docs.page_content)
        except Exception as e:
            print(e)
        


                    

if __name__ == '__main__':
    main()
    
