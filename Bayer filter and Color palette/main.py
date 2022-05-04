import argparse
import numpy as np
import cv2 # OpenCV

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="Lighthouse_bggr.png", type=str, help="Image to load.")
parser.add_argument("--task", default="bayer", type=str, help="Selected task.")

def medianCut(image : np.ndarray, numColors : int) -> list:

    if(image.size <= numColors):
        return image

    if numColors == 1:
        r = int(np.mean(image[:, 0]))
        g = int(np.mean(image[:, 1]))
        b = int(np.mean(image[:, 2]))
        return [[r, g, b]]
    
    r_dif = np.max(image[:, 0]) - np.min(image[:, 0])
    g_dif = np.max(image[:, 1]) - np.min(image[:, 1])
    b_dif = np.max(image[:, 2]) - np.min(image[:, 2])

    rgb_difs = [r_dif, g_dif, b_dif]
    channel = rgb_difs.index(max(rgb_difs))
    image = image[image[:, channel].argsort()]
    median = len(image) // 2

    colors1 = medianCut(image[0:median], numColors // 2)
    colors2 = medianCut(image[median:], numColors // 2)

    return colors1 + colors2


    for r, g, b in palette:
        old_colors = [c for c in new_colors.keys() if (r, g, b) == new_colors[c]]
        mean0 = int(sum(r[0] for r in old_colors)/len(old_colors))
        mean1 = int(sum(g[1] for g in old_colors)/len(old_colors))
        mean2 = int(sum(b[2] for b in old_colors)/len(old_colors))

        cent = (mean0, mean1, mean2)

        if cent != (r, g, b):
            return False
    return True

def bayer(img: np.ndarray) -> np.ndarray:
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    kernel_vertical = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]) / 2
    kernel_horizontal = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]) / 2
    kernel_diagonal = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) / 4
    kernel_full = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4

    #complete red pixels
    green_for_red = cv2.filter2D(green, -1, kernel_full)
    img[::2, ::2, 1] = green_for_red[::2, ::2]
    blue_for_red = cv2.filter2D(blue, -1, kernel_diagonal)
    img[::2, ::2, 2] = blue_for_red[::2, ::2]

    #complete blue pixels
    red_for_blue = cv2.filter2D(red, -1, kernel_diagonal)
    img[1::2, 1::2, 0] = red_for_blue[1::2, 1::2]
    green_for_blue = cv2.filter2D(green, -1, kernel_full)
    img[1::2, 1::2, 1] = green_for_blue[1::2, 1::2]

    #complete green pixels
    red_for_green_odd = cv2.filter2D(red, -1, kernel_vertical)
    img[1::2, ::2, 0] = red_for_green_odd[1::2, ::2]
    red_for_green_even = cv2.filter2D(red, -1, kernel_horizontal)
    img[::2, 1::2, 0] = red_for_green_even[::2, 1::2]
    blue_for_green_odd = cv2.filter2D(blue, -1, kernel_horizontal)
    img[1::2, ::2, 2] = blue_for_green_odd[1::2, ::2]
    blue_for_green_even = cv2.filter2D(blue, -1, kernel_vertical)
    img[::2, 1::2, 2] = blue_for_green_even[::2, 1::2]

    return img

def main(args : argparse.Namespace):

    img = cv2.imread(args.image)
    
    if args.task == "bayer":
        img_result = bayer(img)
        cv2.imshow('result', img_result)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    if args.task == "palette":
        rows, cols = img.shape[0:2]
        num_of_colors = 8
        all_pixels = np.reshape(img, (rows*cols, 3))
        all_colors = np.unique(all_pixels, axis = 0)
        palette = np.array(medianCut(all_colors, num_of_colors))
        print(palette)

    pass


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
