import os
import glob
import imageio
import random, shutil
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as func
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

# import gc

# gc.collect()

# torch.cuda.empty_cache()

def ubah_spect(audio,judul):
  # contoh = 'Data/genres_original/jazz/jazz.00000.wav'
  # y, sample_rate = librosa.load(audio)

  # Plot th sound wave.

  # plt.figure(figsize=(15, 5))
  # librosa.display.waveplot(y=y, sr=sample_rate);
  # plt.title("Sound wave of jazz.00000.wav", fontsize=20)
  # plt.show()

  # Convert sound wave to mel spectrogram.

  y, sr = librosa.load(audio)

  S = librosa.feature.melspectrogram(y, sr=sr)
  S_DB = librosa.amplitude_to_db(S, ref=np.max)
  plt.figure(figsize=(15, 5))
  librosa.display.specshow(S_DB, sr=sr, hop_length=512,x_axis='time', y_axis='log')
  plt.savefig(judul)
  print('tersimpan')


def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
      print("cuda tidak terdeteksi")
  else:
      print("cuda terdeteksi")

  return device


#  Plotting function.

def plot_loss_accuracy(train_loss, train_acc, test_loss, test_acc,judul):
  
  epochs = len(train_loss)
  print(epochs)
  print(train_loss[0].detach().cpu().numpy())
  print(train_loss[0])
  fig, (ax1, ax2) = plt.subplots(1, 2)
  ax1.plot(list(range(epochs)), train_loss, label='Training Loss')
  ax1.plot(list(range(epochs)), test_loss, label='Testing Loss')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.set_title('Epoch vs Loss')
  ax1.legend()

  ax2.plot(list(range(epochs)), train_acc, label='Training Accuracy')
  ax2.plot(list(range(epochs)), test_acc, label='Testing Accuracy')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Accuracy')
  ax2.set_title('Epoch vs Accuracy')
  ax2.legend()
  plt.savefig(judul)
  print('tersimpan')
  # fig.set_size_inches(15.5, 5.5)
  # plt.show()


# Create folder with training, and testing data.

spectrograms_dir = "Data/images_original/"
folder_names = ['Data/train/', 'Data/test/']
train_dir = folder_names[0]
test_dir = folder_names[1]
device = set_device()

for f in folder_names:
  if os.path.exists(f):
    shutil.rmtree(f)
    os.mkdir(f)
  else:
    os.mkdir(f)

# Loop over all genres.

genres = list(os.listdir(spectrograms_dir))
for g in genres:
  # find all images & split in train, and test
  src_file_paths= []
  for im in glob.glob(os.path.join(spectrograms_dir, f'{g}',"*.png"), recursive=True):
    src_file_paths.append(im)
  random.shuffle(src_file_paths)
  test_files = src_file_paths[0:10]
  train_files = src_file_paths[10:]

  #  make destination folders for train and test images
  for f in folder_names:
    if not os.path.exists(os.path.join(f + f"{g}")):
      os.mkdir(os.path.join(f + f"{g}"))

  # copy training and testing images over
  for f in train_files:
    shutil.copy(f, os.path.join(os.path.join(train_dir + f"{g}") + '/',os.path.split(f)[1]))
  for f in test_files:
    shutil.copy(f, os.path.join(os.path.join(test_dir + f"{g}") + '/',os.path.split(f)[1]))

# Data loading.

train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.ToTensor(),
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=10, shuffle=True, num_workers=0)

test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.ToTensor(),
    ]))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=10, shuffle=True, num_workers=0)

def confusion(net,test_loader,judul):
  y_pred = []
  y_true = []

  # iterate over test data
  for data, target in test_loader:

    data, target = data.to(device), target.to(device)
    output = net(data) # Feed Network

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output) # Save Prediction
        
    target = target.data.cpu().numpy()
    y_true.extend(target) # Save Truth
  
  # constant for classes
  classes = ( 'country', 'metal', 'rock', 'disco', 'reggae', 'classical', 'hiphop', 'pop', 'blues', 'jazz')

  # Build confusion matrix
  cf_matrix = confusion_matrix(y_true, y_pred)
  df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
  plt.figure(figsize = (12,7))
  sn.heatmap(df_cm, annot=True)
  plt.savefig(judul)
  print("confusion tersimpan")

def train(model, device, train_loader, test_loader, optimizer, epochs,nilaistop,judul):

  criterion =  nn.CrossEntropyLoss()
  train_loss, test_loss = [], []
  train_acc, test_acc = [], []
  with tqdm(range(epochs), unit='epoch') as tepochs:
    tepochs.set_description('Training')
    for epoch in tepochs:
      model.train()
      # keep track of the running loss
      running_loss = 0.
      correct, total = 0, 0

      for data, target in train_loader:
        # getting the training set
        data, target = data.to(device), target.to(device)
        # Get the model output (call the model with the data from this batch)
        output = model(data)
        # Zero the gradients out
        optimizer.zero_grad()
        # Get the Loss
        loss  = criterion(output, target)
        # Calculate the gradients
        loss.backward()
        # Update the weights (using the training step of the optimizer)
        optimizer.step()

        tepochs.set_postfix(loss=loss.item())
        running_loss += loss  # add the loss for this batch

        # get accuracy
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

      # append the loss for this epoch
      train_loss.append(running_loss/len(train_loader))
      train_acc.append(correct/total)

      # evaluate on test data
      model.eval()
      running_loss = 0.
      correct, total = 0, 0

      # if epoch == nilaistop:
      #   confusion(model,test_loader,judul)
      #   return train_loss, train_acc, test_loss, test_acc

      for data, target in test_loader:
        # getting the test set
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        tepochs.set_postfix(loss=loss.item())
        running_loss += loss.item()
        # get accuracy
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

      test_loss.append(running_loss/len(test_loader))
      test_acc.append(correct/total)

      if epoch == nilaistop:
        checkpoint = {
          "epoch":26,
          "model_state":model.state_dict(),
          "optim_state": optimizer.state_dict()
        }
        torch.save(checkpoint,"modelsaveresnetadam.pth")
        print("model tersimpan")

  return train_loss, train_acc, test_loss, test_acc


# Run training.

#resnet adam
net3 = models.resnet50(pretrained=True)
net3 = net3.to(device)
optimizer3 = torch.optim.Adam(net3.parameters(), lr=0.0001)
judul = "2confresnetadam10100"

train_loss, train_acc, test_loss, test_acc = train(net3, device, train_loader, test_loader, optimizer3, 100,26,judul)
np.save('trainlossresnetadam.npy', train_loss)
np.save('trainaccresnetadam.npy', train_acc)
np.save('testlossresnetadam.npy', test_loss)
np.save('testaccresnetadam.npy', test_acc)
plot_loss_accuracy(train_loss, train_acc, test_loss, test_acc,judul)

#alex adam
net = models.alexnet(pretrained=True)
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
judul = "2confalexadam10100"

train_loss, train_acc, test_loss, test_acc = train(net, device, train_loader, test_loader, optimizer, 100,33,judul)
np.save('trainlossalexadam.npy', train_loss)
np.save('trainaccalexadam.npy', train_acc)
np.save('testlossalexadam.npy', test_loss)
np.save('testaccalexadam.npy', test_acc)
plot_loss_accuracy(train_loss, train_acc, test_loss, test_acc, judul)

#alex SGD
net2 = models.alexnet(pretrained=True)
net2 = net2.to(device)
optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.0001)
judul = "2confalexSGD10100"

train_loss, train_acc, test_loss, test_acc = train(net2, device, train_loader, test_loader, optimizer2, 100,99,judul)
np.save('trainlossalexsgd.npy', train_loss)
np.save('trainaccalexsgd.npy', train_acc)
np.save('testlossalexsgd.npy', test_loss)
np.save('testaccalexsgd.npy', test_acc)
plot_loss_accuracy(train_loss, train_acc, test_loss, test_acc,judul)

#resnet SGD
net4 = models.resnet50(pretrained=True)
net4 = net4.to(device)
judul = "2confresnetSGD10100"
optimizer4 = torch.optim.SGD(net4.parameters(), lr=0.0001)
train_loss, train_acc, test_loss, test_acc = train(net4, device, train_loader, test_loader, optimizer4, 100,80,judul)
np.save('trainlossresnetsgd.npy', train_loss)
np.save('trainaccresnetsgd.npy', train_acc)
np.save('testlossresnetsgd.npy', test_loss)
np.save('testaccresnetsgd.npy', test_acc)
plot_loss_accuracy(train_loss, train_acc, test_loss, test_acc,judul)

# #alex adam no pretrained
# net5 = models.alexnet(pretrained=False)
# net5 = net5.to(device)
# optimizer5 = torch.optim.Adam(net5.parameters(), lr=0.0001)
# judul = "alexadamNOPRETRAINED10100"

# train_loss, train_acc, test_loss, test_acc = train(net5, device, train_loader, test_loader, optimizer5, 100)
# plot_loss_accuracy(train_loss, train_acc, test_loss, test_acc, judul)

# #alex SGD no pretrained
# net6 = models.alexnet(pretrained=False)
# net6 = net6.to(device)
# optimizer6 = torch.optim.SGD(net6.parameters(), lr=0.0001)
# judul = "alexSGDNOPRETRAINED10100"

# train_loss, train_acc, test_loss, test_acc = train(net6, device, train_loader, test_loader, optimizer6, 100)
# plot_loss_accuracy(train_loss, train_acc, test_loss, test_acc,judul)

# #resnet adam no pretrained
# net7 = models.resnet50(pretrained=False)
# net7 = net7.to(device)
# optimizer7 = torch.optim.Adam(net7.parameters(), lr=0.0001)
# judul = "resnetadamNOPRETRAINED10100"

# train_loss, train_acc, test_loss, test_acc = train(net7, device, train_loader, test_loader, optimizer7, 100)
# plot_loss_accuracy(train_loss, train_acc, test_loss, test_acc,judul)

# #resnet SGD no pretrained
# net8 = models.resnet50(pretrained=False)
# net8 = net8.to(device)
# judul = "resnetSGDNOPRETRAINED10100"
# optimizer8 = torch.optim.SGD(net8.parameters(), lr=0.0001)
# train_loss, train_acc, test_loss, test_acc = train(net8, device, train_loader, test_loader, optimizer8, 100)
# plot_loss_accuracy(train_loss, train_acc, test_loss, test_acc,judul)
