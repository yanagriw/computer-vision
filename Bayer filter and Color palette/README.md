# Bayer filter
A Bayer filter mosaic is a color filter array for arranging RGB color filters on a square grid of photosensors. Its particular arrangement of color filters is used in most single-chip digital image sensors used in digital cameras, camcorders, and scanners to create a color image. The filter pattern is half green, one quarter red and one quarter blue, hence is also called *BGGR*.  
![alt text](https://github.com/yanagriw/computer-vision/blob/main/Bayer%20filter%20and%20Color%20palette/Bayer_bggr.png)

<ins>The goal of this function</ins> is to create a colour image out of the image that contains information about intensities in Bayer BGGR pattern (mosaic) by simple averaging in a 3x3 neighbourhood.  
The initial image with intensity information is given by `--image` command-line argument. 

# Color palette
<ins>The goal of this function</ins> is to find a colour palette of an image according to division by median (generally known as *'median cut'*). It is assumed that the image is in RGB colour space.    
The image is provided as `--image` command-line argument. If the image contains fewer colours than the requested number then program will return the original set of colours from the image.
