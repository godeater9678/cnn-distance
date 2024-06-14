import matplotlib.pyplot as plt
import cv2
import numpy as np

def create_chessboard_image(rows, cols, square_size):
    """
    Create a chessboard image with the given number of rows and columns.

    Args:
    - rows (int): Number of rows of squares
    - cols (int): Number of columns of squares
    - square_size (int): Size of each square in pixels

    Returns:
    - image (numpy.ndarray): Generated chessboard image
    """
    # Create a black and white checkerboard image
    board = np.zeros((rows * square_size, cols * square_size), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                cv2.rectangle(board, (j * square_size, i * square_size),
                              ((j + 1) * square_size, (i + 1) * square_size), 255, -1)

    return board


# Parameters
rows = 6
cols = 9
square_size = 50  # Size of each square in pixels

# Create chessboard image
chessboard_image = create_chessboard_image(rows, cols, square_size)

# Save the image
cv2.imwrite('cali.png', chessboard_image)

# Display the image
plt.imshow(chessboard_image, cmap='gray')
plt.axis('off')
plt.show()
