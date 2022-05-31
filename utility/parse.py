import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import itertools
from PIL import Image

def parse_matrix_from_image(filepath, verbose=0):
    """Parse an environment matrix from image file.

    Arguments:
        filepath (str): absolute filepath to target map.

    Returns:
        matrix (numpy.array): formatted array of environment matrix.
    """

    # open file
    image = Image.open(filepath)
    image.load()

    # convert image into numpy array
    image_data = np.asarray(image)

    # convert into heatmap data
    shape = np.shape(image_data)
    matrix = np.zeros(shape, dtype="uint8")

    for (x, y), pixel in np.ndenumerate(image_data):
        pixel = int(pixel)
        matrix[x, y] = pixel
    
    # slightly verbose: display composition of map, plus plot
    if verbose > 0:
        print(Counter(i for i in list(itertools.chain.from_iterable(matrix))))
        plt.imshow(
            matrix,
            cmap=plt.get_cmap("Accent"),
        )
        plt.plot()
    
    # most verbose: write matrix to file to check pixel values
    if verbose > 1:
        with open(f"{filepath}.txt", "w") as file:
            with np.printoptions(threshold=np.inf):
                file.write(str(matrix))

    return matrix


