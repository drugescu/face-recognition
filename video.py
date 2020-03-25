# Initializations
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import keras
import mtcnn.mtcnn
import time
import sys

print("MTCNN version :", mtcnn.__version__)
print("Keras version :", keras.__version__)

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
STATE_RECOGNITION = 2

state = STATE_PASSIVE


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
def extract_face(image, detector_instance, required_size=(112, 112)):
    # Load image
    # image = Image.open(filename)
    # image = image.convert('RGB')
    pixels = np.asarray(image)

    # Detect face
    detector = detector_instance
    results = detector.detect_faces(pixels)

    # If multiple faces in picture, output error
    if (len(results) > 1):
        # print("Detected more than one face!")
        return ["ErrorM"]

    if (len(results) == 0):
        # print("Detected no faces!")
        return ["Error0"]

    # Extract indices of face
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + width

    # Extract face
    face = pixels[y1:y2, x1:x2]

    # Resize to proper dimension
    img_face = Image.fromarray(face)
    img_face = img_face.resize(required_size)
    face_array = np.asarray(img_face)

    return (results, face_array)


def do_verification(face):
    return


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
            state = STATE_RECOGNITION
            return

        # Input processing
        if k == ord('q'):
            sys.exit(0)

    return


# ----------------------------------------------------------------------------------------------------------------------
# Recognition state
# ----------------------------------------------------------------------------------------------------------------------

def recognition_menu():
    global counter
    global state
    global STEPS

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
        face = extract_face(frame, detector, (112, 112))

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
            sleep(2000)
            do_verification(face)
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
# Main body
# ----------------------------------------------------------------------------------------------------------------------

while (True):

    # If state is passive, wait for input on what to do and display standard message
    if state == STATE_PASSIVE:
        # Our operations on the frame come here
        passive_menu()

    # Recognition state
    if state == STATE_RECOGNITION:
        recognition_menu()

    # Input processing
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    sys.exit(0)

# When everything done, release the capture
# cv2.imwrite("result.png", cv2_im_processed)
cap.release()
cv2.destroyAllWindows()
