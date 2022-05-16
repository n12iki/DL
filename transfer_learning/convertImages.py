import sys
from pathlib import Path
import cv2
import numpy as np
import csv

def read_csv(file):
  with open(file, newline='') as csvfile:
    data = list(csv.reader(csvfile))
  return data



def main(imagePath,savePath,labelsPath,labelsToysPath,wordMapPath):
  path=imagePath
  pathlist = Path(path).rglob('*')
  

  #convert jpg images
  count=0
  labels=read_csv(labelsPath)[1:]
  labelsToys=read_csv(labelsToysPath)[1:]
  wordMap=read_csv(wordMapPath)
  words={}
  for i in wordMap:
    words[i[1].replace(" ", "")]=int(i[0])
  labelsOverall={}
  #print(words)
  for i in labels:
    labelsOverall[i[0]]=words[i[1].replace(" ", "")]
  for i in labelsToys:
    labelsOverall[i[0]]=words[i[1].replace(" ", "")]
  #print(labelsOverall)

  
  for path in pathlist:
    if count%2000==0:
      print()
      print(count/37400)
    count=count+1
    img=cv2.imread(str(path))
    #print(str(path)[len(imagePath):-4])
    #print([img,labelsOverall[(str(path)[len(imagePath)+1:])]])
    if(str(path)[-4:]==".jpg"):
        np.save(savePath+str(path)[len(imagePath):-4],np.array([img,labelsOverall[(str(path)[len(imagePath)+1:])]],dtype=object))
    elif(str(path)[-5:]==".jpeg"):
        np.save(savePath+str(path)[len(imagePath):-5],np.array([img,labelsOverall[(str(path)[len(imagePath)+1:])]],dtype=object))

if __name__ == "__main__":
  try:
    images=sys.argv[1]
    save=sys.argv[2]
  except:
    images="dataset/data_256"
    save="dataset/numpy_256"
    labels="MAMe_metadata/MAMe_dataset.csv"
    labelsToys="MAMe_metadata/MAMe_toy_dataset.csv"
    wordMap="MAMe_metadata/MAMe_labels.csv"


  main(images,save,labels,labelsToys,wordMap)