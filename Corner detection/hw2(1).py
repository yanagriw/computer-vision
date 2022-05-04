import argparse
import numpy as np
import scipy
import hw2help
import skimage
import cv2
from numpy.linalg import eig

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="corner2.gif", type=str, help="Image to load.")
parser.add_argument("--task", default="corners", type=str, help="Selected task: corners/susan.")
parser.add_argument("--response", default="harris and stephens", type=str, help="Response function for 1st task.")

NEIGHBOURS = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
SUSAN_MASK = [(x,y) for x in range(-3, 4) for y in range(-3, 4) if (x,y) not in [(2,3),(3,2),(3,3),(2,-3),(3,-2),(3,-3),(-2,3),(-3,2),(-3,3),(-2,-3),(-3,-2),(-3,-3)]]

def get_neighbors(rows, columns, mask):
    neighbors = np.ndarray((rows,columns), dtype=object)
    for i in range(rows):
        for j in range(columns):
            neighbors[i,j] = [ (i+x,j+y) for (x,y) in mask if 0 <= i+x and i+x < rows and 0 <= j+y and j+y < columns ]
    return neighbors

def harris_and_stephens(e1, e2):
    return e1 * e2 - 0.04 * pow( ( e1 + e2 ), 2 )

def shi_and_tomasi(e1, e2):
    return min(e1, e2)

def triggs(e1, e2):
    return min(e1, e2) - 0.04 * max(e1, e2)

def brown_szeliski_winder(e1, e2):
    return (e1 * e2) / (e1 + e2)

def responseCorners(args : argparse.Namespace):
    img = skimage.io.imread(args.image)
    neighbors = get_neighbors(img.shape[0], img.shape[1], NEIGHBOURS)
    img_g = skimage.filters.gaussian(img, sigma=1, truncate=2)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    Dx = cv2.Sobel(img_g, ddepth=-1, dx=1, dy=0, ksize=3)
    Dy = cv2.Sobel(img_g, ddepth=-1, dx=0, dy=1, ksize=3)
    Dx2 = skimage.filters.gaussian(Dx*Dx, sigma=1, truncate=2)
    Dy2 = skimage.filters.gaussian(Dy*Dy, sigma=1, truncate=2)
    Dxy = skimage.filters.gaussian(Dx*Dy, sigma=1, truncate=2)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):

            A = np.matrix( [[Dx2[x,y], Dxy[x,y]], 
                            [Dxy[x,y], Dy2[x,y]]] )
            for (xi, yi) in neighbors[x, y]:
                A = np.add( A, [[Dx2[xi,yi], Dxy[xi,yi]], 
                                [Dxy[xi,yi], Dy2[xi,yi]]] )

            e1, e2 = eig(A)[0]
            if(args.response == "harris and stephens"):
                e = harris_and_stephens(e1, e2)
            elif(args.response == "shi and tomasi"):
                e = shi_and_tomasi(e1, e2)
            elif(args.response == "triggs"):
                e = triggs(e1, e2)
            elif(args.response == "brown szeliski winder"):
                e = brown_szeliski_winder(e1, e2)
            
            if e > 0.2:
                cv2.circle( img, (y, x), 3, (0,0,255), -1)

    return img

def susan(args : argparse.Namespace):
    img = skimage.io.imread(args.image)
    output = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    neighbours = get_neighbors(img.shape[0], img.shape[1], SUSAN_MASK)

    for x in range(img.shape[0]):
        for y in range (img.shape[1]):
            usan = 0
            for (xi, yi) in neighbours[x, y]:
                if img[xi,yi] == img [x,y]:
                    usan += 1
            if usan < 18.5 and usan > 7:
                cv2.circle( output, (y, x), 3, (0,0,255), -1)
    return output

def main(args : argparse.Namespace):

    if args.task == "corners":
        img = responseCorners(args)
        cv2.imshow('corners', img)
        cv2.waitKey(0)

    if args.task == "susan":
        img = susan(args)
        cv2.imshow('corners', img)
        cv2.waitKey(0)

    # TODO (Task 3): Prove the equality with gaussian derivations attached to the homework.
    # - Write the proof on a piece of paper (and submit its picture) or in LaTeX.
    # - Please, don't write the proof in this source file.
    #
    # NOTE: This task does not require any programming - it is a pen&paper exercise.
    pass


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
