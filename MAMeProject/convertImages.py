import sys
from pathlib import Path
import cv2
import numpy as np

def main(imagePath,savePath):
  path=imagePath
  pathlist = Path(path).rglob('*')
  
  #convert jpg images
  count=0
  for path in pathlist:
    if count%2000==0:
      print(count/37400)
    count=count+1
    img=cv2.imread(str(path))
    #print(str(path)[len(imagePath):-4])
    if(str(path)[-4:]==".jpg"):
        np.save(save+str(path)[len(imagePath):-4],img)
    elif(str(path)[-5:]==".jpeg"):
        np.save(save+str(path)[len(imagePath):-5],img)

if __name__ == "__main__":
  try:
    images=sys.argv[1]
    save=sys.argv[2]
  except:
    images="../../../MAMe_data_256/data_256"
    save="../../../MAMe_data_256/numpy_256"

  main(images,save)