import cv2
import numpy as np

path = 'C:/Users/bsdel/PycharmProjects/CSC580/images/Elon_Musk.jpg'


def resize_img():
    # Window name in which image is displayed
    window_name = 'Resized Image'

    # read image
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # #percent by which the image is resized
    scale_percent = 100
    #
    # #calculate the 50 percent of original dimensions
    # width = int(src.shape[1] * scale_percent / 100)
    # height = int(src.shape[0] * scale_percent / 100)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(image, dsize)

    # write image to file
    # cv2.imwrite...

    return output


def print_img_size(image):
    # using imread()
    img = cv2.imread(path)

    # get dimensions of image
    dimensions = img.shape

    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    print('Image Dimension    : ', dimensions)
    print('Image Height       : ', height)
    print('Image Width        : ', width)
    print('Number of Channels : ', channels)


if __name__ == "__main__":
    # print_img_size(path)

    # Resize image
    resized_image = resize_img()
    print_img_size(resized_image)

    # Draw rectrangle
    cv2.rectangle(resized_image, (295, 330), (145, 172), (0, 0, 255), 3)


    # Tag the image
    cv2.putText(resized_image, "CSC580 Wk1", (240, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 4)

    cv2.imshow('Image of Elon', resized_image)

    # Wait for input
    cv2.waitKey(0)

    # kill all windows
    cv2.destroyAllWindows()