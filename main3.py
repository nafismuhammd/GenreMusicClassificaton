import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import librosa
import streamlit as sl
import torchvision.datasets as datasets
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
      print("cuda tidak terdeteksi")
  else:
      print("cuda terdeteksi")

  return device

device = set_device()


def ubah_spect(audio):

  y, sr = librosa.load(audio)
#
  S = librosa.feature.melspectrogram(y=y, sr=sr,n_fft = 2048,hop_length=512,win_length=None,window='hann',center=True,pad_mode='reflect',power=2.0,n_mels=128)
  S_DB = librosa.amplitude_to_db(S, ref=np.max)
  plt.figure(figsize=(15, 5))
  librosa.display.specshow(S_DB, sr=sr, hop_length=512)
  plt.savefig("tes1.png")
  print('tersimpan')

# audio = "hiphop.00018.wav"
# judul = "tes1"
# ubah_spect(audio,judul)


def prediksi():
  with torch.no_grad():
    img = Image.open("tes1.png").convert("RGB")
    img = img.resize((432,288))
    convert_tensor = transforms.ToTensor()
    img = convert_tensor(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    model = models.resnet50(pretrained=True)
    model = model.to(device)

    model.load_state_dict(torch.load("C:/Users/Administrator/PycharmProjects/skripsi/modelsaveresnetadam.pth"))
    # model.load_state_dict(torch.load("modelsaveresnetadam.pth"), strict=False)
    model.eval()


    output = model(img)
    _,pred = torch.max(output.data,1)
    if pred == 0 :
      pred = "country"
    if pred == 1 :
      pred = "metal"
    if pred == 2 :
      pred = "rock"
    if pred == 3 :
      pred = "disco"
    if pred == 4 :
      pred = "reggae"
    if pred == 5 :
      pred = "classical"
    if pred == 6 :
      pred = "hiphop"
    if pred == 7 :
      pred = "pop"
    if pred == 8 :
      pred = "blues"
    if pred == 9 :
      pred = "jazz"
    print(pred)
    return pred

# print(prediksi())