import cv2
import numpy as np
from matplotlib import pyplot as plt

def readIm(filename, flagColor=1):
  im = np.asarray(cv2.imread(filename, flagColor), dtype=np.float32)
  if flagColor == 1:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) #Usaremos de forma consistente el esquema RGB
  return im

def rangeDisplay01(im, flag_GLOBAL):
  #check image type (grayscale or color)
  #im = im.astype(np.float64)
  if len(np.shape(im)) == 2: #Las dimensiones indican que tiene un sólo canal y es grayscale
    # normalize the grayscale image
    min = np.min(im)
    max = np.max(im)
    range = max-min
    if range != 0:
      im = (im-min)/range
    else:
      im = 0
  # compute range and apply normalization
  else:
    if flag_GLOBAL:
      min = np.min(im)
      max = np.max(im)
      range = max-min
      if range != 0:
        im = (im-min)/range
      else:
        im = 0
    else:
      min = np.min(im, axis=(0, 1))
      max = np.max(im, axis=(0, 1))
      range = max-min
    # normalize each band as a grayscale image 
      for i in np.arange(3):
        if range[i] != 0:
          im[i] = (im[i]-min[i])/range[i]
        else:
          im[i] = 0

  return im


def displayIm(im, title='',factor= 1, showFlag=True):  
  # Normalize range
  im = rangeDisplay01(im, 0)
  # Display the image
  if len(im.shape) == 3:
    # im has three channels
    plt.imshow(im.astype('uint8'))
  else:
    # im has a single channel
    plt.imshow(im, cmap='gray')
  figure_size = plt.gcf().get_size_inches()
  plt.gcf().set_size_inches(factor * figure_size)
  plt.title(title) #adding title
  plt.xticks([]), plt.yticks([]) #axis label off
  if showFlag: plt.show()






def displayMI_ES(vim, title='',factor=1, ixf=4):
  # Let's start with case (a). We concatenate the images by columns, or by rows 
  # and columns, depending on the number of images and their dimensions
  vim = np.array(vim)
  nim = len(vim)
  resto = nim % ixf
  fil = nim // ixf

  if resto!=0:
    ultimas = cv2.hconcat(vim[nim-resto:nim])                   #Separamos las últimas imágenes que necesitan rellenar huecos
    primeras = np.split(vim[0:nim-resto], fil) if fil!=0 else []#Dividimos en filas las imágenes
    filas = [cv2.hconcat(a) for a in primeras]                  #Concatenamos cada fila
    if nim>ixf:                                                 #En caso de que haya 1 sóla fila no es necesario rellenar con negro
      for i in range(ixf-resto):
        ultimas = cv2.hconcat([ultimas, np.zeros_like(vim[0])]) #Rellenamos los huecos con color negro
    filas.append(ultimas)
    out = cv2.vconcat(filas)                                    #Concatenamos las filas verticalmente
  else:
    primeras = np.split(vim, fil)
    filas = [cv2.hconcat(a) for a in primeras]
    out = cv2.vconcat(filas)

  return displayIm(out,title,factor)
  
def tam(im):
  return im.shape[0]*im.shape[1]
def displayMI_NES(vim, rows=2):
  #Convert grayscale to 3-channel images if there's incongruence
  if np.any(np.array([True if len(im.shape)==3 else False for im in vim])):
    for im in vim:
      if len(im.shape)==2:
        cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
  #Sort the vim in size order
  vim.sort(key=tam)
  vim.reverse()

  #Calculamos la forma de la cuadrícula en función de las imágenes
  nim = len(vim)
  imxfila = int(nim/rows + 0.5)
  imxcolumn = int(nim/imxfila+0.5)

  #Calculamos los máximos de las anchuras y alturas para poder rellenar los huecos posteriormente
  mxr = np.zeros(imxfila, dtype=int)
  for i in range(nim):
    mxr[i//imxcolumn] = max(mxr[i//imxcolumn], vim[i].shape[0])
  maxrow = np.sum(mxr)
  mxc = np.zeros(imxcolumn, dtype=int)
  for i in range(nim):
    mxc[i%imxcolumn] = max(mxc[i%imxcolumn], vim[i].shape[1])
  maxcolumn = np.sum(mxc)

  black = np.zeros(len(vim[-1].shape))  #Color del borde
  for i in range(len(vim)): #En cada imagen ajustamos los huecos para que sean del mismo tamaño y podamos reducirnos al caso a)
    vim[i] = cv2.copyMakeBorder(vim[i], 0, mxr[i//imxcolumn]-vim[i].shape[0], 0, mxc[i%imxcolumn]-vim[i].shape[1], cv2.BORDER_CONSTANT, value=black)

#Creamos las filas con las imagenes en el nuevo orden
  canvas = []
  for i in range(imxfila):
    canvas.append(vim[imxcolumn*i:imxcolumn*(i+1)])


  outIm = cv2.hconcat(canvas[0])  #Apilamos las imágenes con bordes como en el primer apartado 
  for i in range(1, len(canvas)):
    row = cv2.hconcat(canvas[i])
    outIm = cv2.vconcat([outIm, row])
  return outIm[0:maxrow,0:maxcolumn, :]

def changePixelValues(im,cp,nv):
  # cp is a vector of pixel coordinates
  # nv is a vector with the new values
  # replace the values of cp with the nv values
  for i, p in enumerate(cp):
    im[p[0]][p[1]] = nv[i]
  return displayIm(im)

def print_images_titles(vim, titles=None, rows=2):
  nim = len(vim)
  imxfila = int(nim/rows + 0.5)
  for i in range(nim):
    plt.subplot(rows, imxfila, i+1)
    if titles:
      plt.title(titles[i])
    displayIm(vim[i], title=titles[i], showFlag=False)
  plt.show()