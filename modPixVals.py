import os, sys, glob, time, argparse, multiprocessing
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--images', '-i', help="full-path to 3 channel image(s)")
parser.add_argument('--annotations', '-a', help="full-path to 1 channel annotation image(s)")
parser.add_argument('--which', '-w', help="which dataset to tile, valid answers: images or annotations")
parser.add_argument('--outPath', '-c', help="full-path to output tiles")

args = parser.parse_args()
print args

images=args.images
annotations=args.annotations
which=args.which
outPath=args.outPath


if not os.path.exists(outPath):
    os.mkdir(outPath)

def modVals(f):
    filename_split = os.path.splitext(f)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    out=outPath+basename[:-4]   #make a directory to write to
    im=np.array(Image.open(f))
    im[im==1]=0
    im[im==2]=1
    im[im==3]=2
    im[im==4]=3
    im[im==5]=4
    im[im==6]=5
    im[im==7]=6
    im[im==8]=7
    im[im==9]=8
    im[im==10]=9
    im[im==11]=10
    im[im==12]=11
    im[im==13]=12
    im[im==14]=13
    im[im==15]=14
    im[im==16]=15
    im[im==17]=16
    im[im==18]=17
    im[im==19]=18
    im[im==20]=19
    im[im==21]=20
    im[im==22]=21
    im[im==23]=22
    im[im==24]=23
    im[im==25]=24
    im[im==26]=25
    size=im.shape
    y=size[0]
    x=size[1]
    im = Image.fromarray(im)
    im.save(out+'.png')
    print(np.unique(im))
    return

def modValsAgnostic(f):
    filename_split = os.path.splitext(f)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    out=outPath+basename[:-4]   #make a directory to write to
    im=np.array(Image.open(f))
    for i in np.unique(im):
    	j = i-1
    	im[im==i]=j
    im = Image.fromarray(im)
    im.save(out+'.png')
    print(np.unique(im))
    return


p = multiprocessing.Pool(30)

if which=='images':
  for f in glob.glob(images+'*.tif'):
      p.apply_async(modValsRGB, [f])
  print('done')
elif which=='annotations':
  for f in glob.glob(annotations+'*.png'):
      p.apply_async(modValsAgnostic, [f])
  print('done')
else:
   print("wrong options, please choose either images or annotations")
   
p.close()
p.join()
