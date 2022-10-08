# Step 1: Import the packages
from keras.models import model_from_json
import cv2
import numpy as np

# Step 2: Load the Model from Json File
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Step 3: Load the weights
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Step 4: Compile the model
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Step 5: load the image you want to test
image = cv2.imread('C:/Users/bsdel/PycharmProjects/CSC580/wk6/cats_and_dogs_filtered/train/dogs/dog.1.jpg')
image2 = cv2.resize(image, (50,50))
image3 = image2.reshape(1, 50, 50, 3)

cv2.imshow("Input Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 6: Predict to which class your input image has been classified
def displayMenu():
    print("==== Select a number from 1-100 for Dog or Cat Image ====\n")
    return int(input("Selection: "))

_uselection = True
while _uselection is True:
    selection = displayMenu()
    if selection > 0:
        s1="C:/Users/bsdel/PycharmProjects/CSC580/wk6/cats_and_dogs_filtered/train/dogs/dog."
        s2=".jpg"
        s3=s1+str(selection)+s2
        image = cv2.imread(s3)

        image2 = cv2.resize(image, (50, 50))
        image3 = image2.reshape(1, 50, 50, 3)

        cv2.imshow("Input Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("")
        print("Invalid number, Please select a number from 1-100")
        print("")

result = loaded_model.predict_classes(image3)

if(result[0][0] == 1):
    print("I guess this must be a Dog!")
else:
    print("I guess this must be a Cat!")