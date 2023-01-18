import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2, numpy as np, math

# We start by getting access to the drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
import os, sys

path_to_module='/content/drive/My Drive/CV/'
sys.path.append(os.path.abspath(path_to_module))

import P0

get_image = lambda route: os.path.join('/content/drive/MyDrive/images', route)

def gaussianMask1D(sigma=0, sizeMask=0, order=0):
  #calcular el otro sigma o mask
  if sigma > 0:
    k = int(sigma*3+1)
  else:
    if sizeMask > 0:
      k = (sizeMask - 1)/2
      sigma = k/3.0

  if order==0:
    mask = [np.exp(-x*x/(2*sigma*sigma)) for x in np.arange(-k, k+1, 1)]
    mask = mask / np.sum(mask)
  if order==1:
    mask = [np.exp(-x*x/(2*sigma*sigma))* (-x/sigma) for x in np.arange(-k, k+1, 1)]
  if order==2:
    mask = [np.exp(-x*x/(2*sigma*sigma))* (x**2/sigma**2 - 1) for x in np.arange(-k, k+1, 1)]

  return np.array(mask)


def plotGraph(graph, title='No title'):
  k = (len(graph)-1)/2
  plt.plot(np.arange(-k, k+1, 1), list(graph))
  plt.title(title)
  plt.show()


def my2DConv(im, sigma, orders):
  if orders==[0, 0]:
    return cv2.sepFilter2D(im, ddepth=cv2.CV_64F, kernelX=gaussianMask1D(sigma=sigma, order=0), kernelY=gaussianMask1D(sigma=sigma, order=0))
  elif orders==[1, 0]:
    return cv2.sepFilter2D(im, ddepth=cv2.CV_64F, kernelX=gaussianMask1D(sigma=sigma, order=1), kernelY=gaussianMask1D(sigma=sigma, order=0))
  elif orders==[0, 1]:
    return cv2.sepFilter2D(im, ddepth=cv2.CV_64F, kernelX=gaussianMask1D(sigma=sigma, order=0), kernelY=gaussianMask1D(sigma=sigma, order=1))
  elif orders==[2,0]:
    return cv2.sepFilter2D(im, ddepth=cv2.CV_64F, kernelX=gaussianMask1D(sigma=sigma, order=2), kernelY=gaussianMask1D(sigma=sigma, order=0))
  elif orders==[0, 2]:
    return cv2.sepFilter2D(im, ddepth=cv2.CV_64F, kernelX=gaussianMask1D(sigma=sigma, order=0), kernelY=gaussianMask1D(sigma=sigma, order=2))
  elif orders==[2, 2]:
    return sigma**2*(my2DConv(im, sigma, [2, 0])+my2DConv(im, sigma, [0, 2]))
  else:
    ('error in order of derivative')


def gradientIM(im,sigma):
  dx = my2DConv(im, sigma, [1, 0])  
  dy = my2DConv(im, sigma, [0, 1])
  return dx,dy

def laplacianG(im,sigma):
  return my2DConv(im, sigma, [2, 2])


def sizetoSigma(sizemask):  #Recuperamos la conversión de tamaño de máscara a sigma que se incluía en otra función
  k = (sizemask - 1)/2
  sigma = k/3.0
  return sigma
    
def pyramidGauss(im,sizeMask=7, nlevel=4):
  sigma = sizetoSigma(sizeMask)                               #Calculamos la desviación con la función
  vim = [im]
  alisar = gaussianMask1D(0, sizeMask, 0)                     #Preparamos la máscara de alisamiento
  for i in range(nlevel):
    im = cv2.sepFilter2D(im, -1, alisar, alisar)[::2, ::2]    #En cada nivel vamos aplicando alisamiento gaussiano y reduciendo el tamaño haciendo subsampling
    vim.append(im)
  return vim                                                  #Devolvemos el vector con las imágenes de la pirámide
  
a = get_image('zebra.jpg')
im = P0.readIm(a, 0)
pyrG = pyramidGauss(im,sizeMask=7, nlevel=4)

def displayPyramid(vim, title='Gaussian Pyramid'):
  maxwidth = vim[1].shape[1]                                  #Calculamos las dimensiones que tendrá el canvas completo
  maxheight = vim[0].shape[0]
  black = np.zeros(len(vim[-1].shape))                        #Generamos el color de relleno para los huecos
  resto = [vim[1]]                                            #Empezamos a definir la parte derecha de la imagen con la segunda imagen de la pirámide
  for i in vim[2:]:
    resto.append(cv2.copyMakeBorder(i, 0, 0, 0,maxwidth-i.shape[1], cv2.BORDER_CONSTANT, value=black))      #Para el resto rellenamos el hueco que falta calculandolo respecto a los máximos
  resto = cv2.vconcat(resto)                                                                                #Concatenamos la columna de la parte derecha
  resto = cv2.copyMakeBorder(resto, 0, maxheight-resto.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=black)    #Rellenamos el hueco hacia abajo con respecto a la primera imagen
  P0.displayIm(np.hstack((vim[0], resto)))



def pyramidLap(im, sizeMask,nlevel=4,flagInterp=cv2.INTER_LINEAR):
  vim = pyramidGauss(im, sizeMask, nlevel)                                                                    #Calculamos la pirámide gaussiana
  vimL = []
  for i in range(0, len(vim)-1):                                                                              
    vimL.append(vim[i] - cv2.resize(vim[i+1], (vim[i].shape[1], vim[i].shape[0]), interpolation=flagInterp))  #En cada nivel, tomamos el mismo de la gaussiana y le restamos el siguiente expandido con interpolación lineal
  vimL.append(vim[nlevel])                                                                                    #El último nivel es el de la gaussiana

  return vimL

def reconstructIm(pyL,flagInterp):
  inverse = list(reversed(pyL))                                                                                        #Invertimos la laplaciana para iterar más comodamente 
  aux = inverse[0]                                                                                                  
  for i in range(0, len(pyL)-1):  
    aux = inverse[i+1] + cv2.resize(aux, (inverse[i+1].shape[1], inverse[i+1].shape[0]), interpolation=flagInterp)     #A partir del ultimo nivel vamos calculando la suma del penúltimo y la reducción a su tamaño del último

  return aux


