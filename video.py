# ----------------------------------------------------------------------------------------------------------------------
# Face detector/recognizer
# ---------------------------------------------------------------------------------------------------------------------

# Starting point: https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

# Used also my own .ipynb files as starting points

# ----------------------------------------------------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import keras
from keras.models import load_model
import mtcnn.mtcnn
import time
import sys
import cryptography
from cryptography.fernet import Fernet
import os.path

print("MTCNN version :", mtcnn.__version__)
print("Keras version :", keras.__version__)

# ---------------------------------------------------------------------------------------------------------------------
# Cryptography
# ---------------------------------------------------------------------------------------------------------------------

# Generate a new key - do this once, write it down, and store it here ecrypted - base64 encoded key
key = b'nP2H4k_dl2-SNeF5TpLALSbgsHBxlmpouZmriC92748='
#key = Fernet.generate_key()
print(key)

# Encryption is done like this
#message = "my deep dark secret".encode()
#f = Fernet(key)
#encrypted = f.encrypt(message)

# Decryption is done like this
#encrypted = b"...encrypted bytes..."
#f = Fernet(key)
#decrypted = f.decrypt(encrypted)

# Cryptographic key should be store in some other ECU of the car and shared over wire on demand, tied to vehicle ID perhaps
# Ideally, car should connect to the internt to allow face recognition and disallow it otherwise. And store key on a server
#   accessible with vehicle ID. Face Recognition requests should then be logged and possible SMS sent.

# ---------------------------------------------------------------------------------------------------------------------
# Embeddings file
# ---------------------------------------------------------------------------------------------------------------------

if os.path.exists("embeddings.npz"):
    embeddings_file = np.load("embeddings.npz")

    # Size
    identities = np.asarray(list(embeddings_file['ids']))
    size_ids = len(embeddings_file[embeddings_file.files[0]])
    if size_ids == 1:
        print ("There is 1 identity recorded in the database")
    else:
        print ("There are ", size_ids, " identities recorded in the database.")
    if size_ids == 0:
        print("Error! No identities in file!")
        sys.exit(0)

    for i in range(size_ids):
        size_emb = len(embeddings_file[embeddings_file.files[0]][i])
        print("There are ", size_emb, " embeddings recorded in the database for ID ", i + 1)
        #print(identities[identities.files[0]][i])

    # Close file
    embeddings_file.close()

    # Print numpy array of embeddings and identities
    print(identities)
else:
    identities = []
    print ("There are no embeddings recorded in the database.")

# Threshold
thresholds = 0.5#np.arange(1, -1, 0.01)
print("Threshold = ", thresholds)
sys.exit(0)

# Load facenet model
class model_class():
    def __init__(self):
        return

    inputs = 0
    outputs = 0

model = model_class()

model = load_model('keras-facenet/model/facenet_keras.h5')
print("Loaded keras-facenet.h5 model")

# Variables
THICKNESS = 10
THICKNESS_FACEBOX = 1
STEPS = 3
counter = -STEPS
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

# Colors
gray = (100, 100, 100)
white = (255, 255, 255)

# Number of frames required for detection
DETECTION_FRAMES = 5

# States of program
STATE_PASSIVE = 0
STATE_LEARNING = 1
STATE_DETECTION = 2
STATE_VERIFICATION = 3

state = STATE_PASSIVE

face = 0


# ----------------------------------------------------------------------------------------------------------------------
# Useful functions
# ----------------------------------------------------------------------------------------------------------------------

# Draw text - expects cv2 image, returns cv2 image
def pil_text(image, x, y, font, size, color, text):
    # Convert image
    cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)

    # Choose a font
    font = ImageFont.truetype(font, size)

    # Draw the text
    draw.text((x, y), text, font=font, fill=color)

    # Save the image
    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    image = cv2_im_processed

    return image


# Draw text with shadow behind
def pil_text_shadow(image, x, y, font, size, color, text):
    image = pil_text(image, x - 1, y - 1, font, size, gray, text)
    image = pil_text(image, x + 1, y + 1, font, size, gray, text)
    image = pil_text(image, x, y, font, size, color, text)
    return image


# Extract face from an image
def extract_face(image, detector_instance, required_size=(160, 160)):
    # Load detector
    # global detector

    # Load image
    # image = Image.open(filename)
    # image = image.convert('RGB')
    pixels = np.asarray(image)

    # Detect face
    det = detector_instance
    results = det.detect_faces(pixels)

    # If multiple faces in picture, output error
    if len(results) > 1:
        # print("Detected more than one face!")
        return ["ErrorM"]

    if len(results) == 0:
        # print("Detected no faces!")
        return ["Error0"]

    # Extract indices of face
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # Extract face
    local_face = pixels[y1:y2, x1:x2]

    # Resize to proper dimension
    img_face = Image.fromarray(local_face)
    img_face = img_face.resize(required_size)
    face_array = np.asarray(img_face)

    return results, face_array


# ----------------------------------------------------------------------------------------------------------------------

# Load MTCNN detector
detector = mtcnn.MTCNN()

# Capture frame
cap = cv2.VideoCapture(0)


# ----------------------------------------------------------------------------------------------------------------------
# Passive state - Main Menu
# ----------------------------------------------------------------------------------------------------------------------

def passive_menu():
    global counter
    global state
    global STEPS

    # Adjust number of steps
    STEPS = 5

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        image = frame

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

        # Draw other stuff - BGR
        counter = counter + 1

        if counter > STEPS:
            counter = -STEPS

        # Draw Text PIL Image
        image = cv2.rectangle(image, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), white, THICKNESS)
        image = pil_text_shadow(image, 150, 20, "Ubuntu-B.ttf", 25, white, "FACE RECOGNITION SYSTEM ")
        image = pil_text_shadow(image, 228, 50, "Ubuntu-L.ttf", 20, white, "R - Register new face")
        image = pil_text_shadow(image, 212, 70, "Ubuntu-L.ttf", 20, white, "U - Unlock using your face")

        # Show image
        cv2.imshow('frame', image)

        # Resize window
        cv2.resizeWindow('frame', 1920, 1080 - 30)

        # Get input
        k = cv2.waitKey(1) & 0xFF
        if k == ord('u'):
            if len(identities) > 0:
                state = STATE_DETECTION
                return

        if k == ord('r'):
            state = STATE_LEARNING
            return

        # Input processing
        if k == ord('q'):
            sys.exit(0)

    return


# ----------------------------------------------------------------------------------------------------------------------
# Detection state
# ----------------------------------------------------------------------------------------------------------------------

def detection_menu():
    global counter
    global state
    global STEPS
    global face

    # Adjust number of steps
    STEPS = 2

    # Must hold single face steady for a number of frames
    timer_tick = 0

    # Retain old image
    # Capture frame-by-frame
    ret, frame = cap.read()
    old_image = frame

    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()
        image = frame

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

        # Draw other stuff - BGR
        if timer_tick < DETECTION_FRAMES:
            counter = counter + 1

        if counter > STEPS:
            counter = -STEPS

        if counter < 0:
            image = cv2.rectangle(image, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), white, THICKNESS)
            image = pil_text_shadow(image, 210, 20, "Ubuntu-B.ttf", 25, white, "DETECTING FACE...")

        # Draw Text PIL Image
        image = pil_text_shadow(image, 209, 50, "Ubuntu-L.ttf", 20, white, "Face and look into camera.")
        image = pil_text_shadow(image, 190, 70, "Ubuntu-L.ttf", 20, white, "Only one face should be visible.")
        image = pil_text_shadow(image, 228, 450, "Ubuntu-L.ttf", 20, white, "B - Back, Q - Quit")

        # Draw detections
        face = extract_face(frame, detector)

        if face != ["ErrorM"] and face != ["Error0"]:
            box = face[0][0]["box"]
            if box is not None:
                # print(detections)
                timer_tick = timer_tick + 1
                image = cv2.rectangle(image,
                                      (box[0], box[1]),
                                      (box[0] + box[2], box[1] + box[3]),
                                      white, THICKNESS_FACEBOX)
            else:
                timer_tick = 0
        else:
            timer_tick = 0

        # If detection complete, show old image
        if timer_tick >= DETECTION_FRAMES:
            old_image = pil_text_shadow(old_image, 228, 420, "Ubuntu-L.ttf", 30, white, "Face detected")
            cv2.imshow('frame', old_image)

            state = STATE_VERIFICATION
            return
        else:
            # Show image
            cv2.imshow('frame', image)
            # Retain old image
            old_image = image

        # Resize window
        cv2.resizeWindow('frame', 1920, 1080 - 30)

        # Get input
        k = cv2.waitKey(1) & 0xFF
        if k == ord('b'):
            state = STATE_PASSIVE
            return

        # Input processing
        if k == ord('q'):
            sys.exit(0)

    return


# ----------------------------------------------------------------------------------------------------------------------
# Verification state
# ----------------------------------------------------------------------------------------------------------------------

def normalize(current_face):
    # Scale pixel values
    face_pixels = current_face.astype('float32')

    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    return face_pixels


def get_embedding(current_face):
    # First normalize
    norm_face = normalize(current_face)

    # Expand dimensions into samples, for now only one
    samples = np.expand_dims(norm_face, axis = 0)

    # Make prediction to get embedding
    yhat = model.predict(samples)

    # Return embedding
    return yhat[0]


def verify(current_face):
    # load the model
    # do this at load

    # summarize input and output shape
    print(model.inputs)
    print(model.outputs)

    # Get embedding
    embedding = get_embedding(current_face)
    print("Embedding = ", embedding, ", of size = ", len(embedding))

    # Go through database of identities and embeddings and verify yes or no

    return


# Interactive menu showing access granted or denied
def verification_menu():
    global state

    # Do verification
    verify(face[1])

    # sys.exit(0)

    # Display stuff and enable interactivity
    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Make it all black
        frame[:] = (0, 0, 0)
        image = frame

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

        # Display other stuff
        image = pil_text(image, 170, 210, "Ubuntu-L.ttf", 40, (0, 255, 0), "ACCESS GRANTED")

        # Display face
        y_offset = 250
        x_offset = 250
        image[y_offset:y_offset + face[1].shape[0], x_offset:x_offset + face[1].shape[1]] = face[1]

        # Show image
        cv2.imshow('frame', image)

        # Resize window
        cv2.resizeWindow('frame', 1920, 1080 - 30)

        # Get input
        k = cv2.waitKey(1) & 0xFF
        if k == ord('b'):
            state = STATE_PASSIVE
            return

        # Input processing
        if k == ord('q'):
            sys.exit(0)

    return


# ----------------------------------------------------------------------------------------------------------------------
# Learning State - Register a new identity
# ----------------------------------------------------------------------------------------------------------------------

identities = []

def learning_menu():
    global state
    global counter
    global STEPS
    global face
    STEPS = 3

    messages = [
        "Face camera straight with neutral face",
        "Face camera straight with eyes closed",
        "Face camera slightly sideways",
        "Face camera and smile",
        "If you have glasses, take them off",
        "Turn reading light on and look slightly down",
        ""
    ]

    MAX_MESSAGES  = 5
    current_message = 0
    timer_tick = 0
    new_identity = []

    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()
        image = frame

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

        # Draw other stuff - BGR
        if timer_tick < DETECTION_FRAMES:
            counter = counter + 1

        if counter > STEPS:
            counter = -STEPS

        if counter < STEPS:
            image = cv2.rectangle(image, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), white, THICKNESS)
            image = pil_text_shadow(image, 160, 20, "Ubuntu-L.ttf", 20, white, messages[current_message])
        # Text
        image = pil_text_shadow(image, 248, 430, "Ubuntu-L.ttf", 20, white, "B - Back, Q - Quit")
        image = pil_text_shadow(image, 254, 450, "Ubuntu-L.ttf", 20, white, "T - Take Photo")

        # Draw detections
        face = extract_face(frame, detector)

        if face != ["ErrorM"] and face != ["Error0"]:
            box = face[0][0]["box"]
            if box is not None:
                # print(detections)
                timer_tick = timer_tick + 1
                image = cv2.rectangle(image,
                                      (box[0], box[1]),
                                      (box[0] + box[2], box[1] + box[3]),
                                      white, THICKNESS_FACEBOX)
            else:
                timer_tick = 0
        else:
            timer_tick = 0

        # Show image
        cv2.imshow('frame', image)

        # Resize window
        cv2.resizeWindow('frame', 1920, 1080 - 30)

        # Exit window after performing final processing - encryption on
        if current_message > MAX_MESSAGES:
            # Store embeddings
            print(np.asarray(new_identity))
            identities.append(np.asarray(new_identity))
            new_identity.clear()
            np.savez_compressed('embeddings', ids = np.asarray(identities))
            state = STATE_PASSIVE
            return

        # Get input
        k = cv2.waitKey(1) & 0xFF
        if k == ord('b'):
            state = STATE_PASSIVE
            return

        # Record new photo
        if k == ord('t'):
            if timer_tick >= DETECTION_FRAMES:
                current_face = face[1]

                # Get face embedding
                embedding = get_embedding(current_face)
                #print("Embedding = ", embedding)

                # Append embedding to new identity
                new_identity.append(embedding)
                print("Embeddings size = ", len(new_identity))

                # Increment message and take new photo
                current_message = current_message + 1

                # Queue ROC analysis

        # Input processing
        if k == ord('q'):
            sys.exit(0)

    return


# ----------------------------------------------------------------------------------------------------------------------
# Main body
# ----------------------------------------------------------------------------------------------------------------------

while (True):

    # If state is passive, wait for input on what to do and display standard message
    if state == STATE_PASSIVE:
        passive_menu()

    # Detection state
    if state == STATE_DETECTION:
        detection_menu()

    # Verification state - validate identity and grant access
    if state == STATE_VERIFICATION:
        verification_menu()

    if state == STATE_LEARNING:
        learning_menu()

    # Input processing
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    sys.exit(0)

# When everything done, release the capture
# cv2.imwrite("result.png", cv2_im_processed)
cap.release()
cv2.destroyAllWindows()
