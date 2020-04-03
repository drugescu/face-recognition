# ----------------------------------------------------------------------------------------------------------------------
# Face detector/recognizer
# ---------------------------------------------------------------------------------------------------------------------

# Starting point: https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

# Used also my own .ipynb files as starting points

# Should move on to use tensorflow 2.0

# ----------------------------------------------------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import mtcnn.mtcnn
import time
import sys
import cryptography
from cryptography.fernet import Fernet
import os.path
# import tensorflow as tf
import keras
from keras.models import load_model
import tensorflow as tf
import sklearn
import sklearn.datasets
import sklearn.metrics
from matplotlib import pyplot as plt
import pandas as pd

# tf.enable_eager_execution()

print("MTCNN version :", mtcnn.__version__)
print("Keras version :", keras.__version__)
print("Tensorflow version :", tf.__version__)

# ---------------------------------------------------------------------------------------------------------------------
# Cryptography
# ---------------------------------------------------------------------------------------------------------------------

# Generate a new key - do this once, write it down, and store it here ecrypted - base64 encoded key
key = b'nP2H4k_dl2-SNeF5TpLALSbgsHBxlmpouZmriC92748='
# key = Fernet.generate_key()
print(key)

# Encryption is done like this
# message = "my deep dark secret".encode()
# f = Fernet(key)
# encrypted = f.encrypt(message)

# Decryption is done like this
# encrypted = b"...encrypted bytes..."
# f = Fernet(key)
# decrypted = f.decrypt(encrypted)

# Cryptographic key should be store in some other ECU of the car and shared over wire on demand, tied to vehicle ID perhaps
# Ideally, car should connect to the internt to allow face recognition and disallow it otherwise. And store key on a server
#   accessible with vehicle ID. Face Recognition requests should then be logged and possible SMS sent.

# ---------------------------------------------------------------------------------------------------------------------
# Embeddings file
# ---------------------------------------------------------------------------------------------------------------------

identities = np.asarray([])

if os.path.exists("embeddings.npz"):
    embeddings_file = np.load("embeddings.npz")

    # Size
    identities = np.asarray(list(embeddings_file['ids']))
    size_ids = len(embeddings_file[embeddings_file.files[0]])
    if size_ids == 1:
        print("There is 1 identity recorded in the database")
    else:
        print("There are ", size_ids, " identities recorded in the database.")
    if size_ids == 0:
        print("Error! No identities in file!")
        sys.exit(0)

    for i in range(size_ids):
        size_emb = len(embeddings_file[embeddings_file.files[0]][i])
        print("There are ", size_emb - 1, " embeddings recorded in the database for ID ", i + 1)
        # print(identities[identities.files[0]][i])

    # Close file
    embeddings_file.close()

    # Print numpy array of embeddings and identities
    print(identities)
else:
    identities = np.asarray([])
    print("There are no embeddings recorded in the database.")

# ---------------------------------------------------------------------------------------------------------------------
# Threshold and model
# ---------------------------------------------------------------------------------------------------------------------

# pre-roc dataset pull
# try: "sudo pip install --upgrade certifi" and then if certificate issues continue, import ssl and _create_unverified_context
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Slice was found empirically
print("Loading LFW from sklearn or disk...")
lfw_people = sklearn.datasets.fetch_lfw_people(min_faces_per_person=1, color=True, resize=1.0,
                                               slice_=(slice(80, 188, 3), slice(80, 170, 3)))
print("Loaded LFW dataset from sklearn, size = ", len(lfw_people.images))
print(lfw_people.target[0], ", ", lfw_people.target_names[0], ", image = ", lfw_people.images[0], " of shape = ",
      lfw_people.images[0].shape)
lfw_people.images_resized = []

for i in range(len(lfw_people.images)):
    lfw_people.images_resized.append(cv2.resize(lfw_people.images[i] / 255.0, (160, 160), cv2.INTER_CUBIC))

print("Cropped and resized images to 160x160.")
print(lfw_people.target[0], ", ", lfw_people.target_names[0], ", image = ", lfw_people.images_resized[0],
      " of shape = ", lfw_people.images_resized[0].shape)


# Display the resulting frame
def debugDisplayLFW():
    # Get a portion of the faces
    vstck = np.vstack(lfw_people.images_resized[30:60])

    b = False
    while b is not True:
        # Show window
        cv2.namedWindow('frameTemp', cv2.WINDOW_NORMAL)
        # cv2_im_processed = cv2.cvtColor(lfw_people.images[0]/255.0, cv2.COLOR_RGB2BGR)

        # Show image
        # cv2.imshow('frameTemp', cv2_2)
        cv2.imshow('frameTemp', vstck)

        k = cv2.waitKey(1) & 0xFF
        # Input processing
        if k == ord('q'):
            b = True


# debugDisplayLFW() # uncomment this to view image selection

# Save images
def saveLFW():
    counter_dict = {}
    for i in range(len(lfw_people.images_resized)):
        if (lfw_people.target[i]) not in counter_dict.keys():
            counter_dict[lfw_people.target[i]] = 0

        counter_dict[lfw_people.target[i]] = counter_dict[lfw_people.target[i]] + 1
        cv2.imwrite("./resized/id" + str(lfw_people.target[i]) + "_" + str(counter_dict[lfw_people.target[i]]) + ".png",
                    cv2.cvtColor(lfw_people.images_resized[i] * 255.0, cv2.COLOR_RGB2BGR))

    print("Wrote resized images")


# Call the function to save the resized images
# saveLFW()
# Otherwise load them

# Threshold
testing_thresholds = [0.3, 0.5, 0.65, 0.75]
thresholds = 0.1
print("Thresholds = ", testing_thresholds)


# sys.exit(0)

# Load facenet model
class model_class():
    def __init__(self):
        return

    inputs = 0
    outputs = 0


# model = model_class()

print("Loading keras-facenet.h5 model...")
model = load_model('keras-facenet/model/facenet_keras.h5')
print("Loaded keras-facenet.h5 model")

# ---------------------------------------------------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------------------------------------------------

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

# Number of faces
NUMBER_OF_FACES = 6

# States of program
STATE_PASSIVE = 0
STATE_LEARNING = 1
STATE_DETECTION = 2
STATE_VERIFICATION = 3

state = STATE_PASSIVE

face = 0
FACE_PIXELS = 1
NO_MATCH = -1


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
    samples = np.expand_dims(norm_face, axis=0)

    # Make prediction to get embedding
    yhat = model.predict(samples)

    # Return embedding
    return yhat[0]


def verify(current_face):
    # load the model
    # do this at load
    global identities
    # should have to get threashold from this and use it

    result = NO_MATCH

    # summarize input and output shape
    print(model.inputs)
    print(model.outputs)

    # Get embedding
    embedding = get_embedding(current_face)
    print("[Verification Menu] Veryfying current face (unlock attempt)...")
    print("Embedding = ", embedding, ", of size = ", len(embedding))

    # Go through database of identities and embeddings and verify yes or no
    # Assumption - do not allow two identities of the same person in the database - detect his situation at registration
    for id in range(len(identities)):
        loss = keras.losses.cosine_similarity(
            identities[id],
            embedding,
        )

        losspy = -loss.eval(session=tf.Session())

        print(" Verifying identity ", id)
        # print("   Embedding ", identities[id][emb_id])
        print("   Cosine similarity ", losspy)

        # How many embeddings have thresholds surpassed
        matches = [x for x in losspy if x > thresholds]

        if (len(matches) > 0):
            result = id

    return result


similarities_current = []

# Interactive menu showing access granted or denied
def verification_menu():
    global state
    global face
    global similarities_current

    # Do verification
    match = verify(face[FACE_PIXELS])
    similarities_current = match

    # Temporary
    #roc_analysis()

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
        if (match != NO_MATCH):
            image = pil_text(image, 170, 210, "Ubuntu-L.ttf", 40, (0, 255, 0), "ACCESS GRANTED")
        else:
            image = pil_text(image, 170, 210, "Ubuntu-L.ttf", 40, (255, 0, 0), "ACCESS DENIED")

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

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = sklearn.metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def roc_analysis(current_id_embeddings):
    """Perform ROC analysis on stored lfw_people with current identity embeddings
    Parameters
    ----------
    current_id-embeddings : list of 128-long embeddings (themselves tensors)

    Returns
    -------
    integer type, with optimal cutoff value

    """
    global lfw_people
    global face
    global thresholds
    global testing_thresholds

    colors = ['c', 'g', 'b', 'y']

    # We are fed a number of embeddings for the current face, take the first one
    test_emb = current_id_embeddings[0] # should generalize this to generate all pairs and run ROC on all
    rest_emb = np.array(current_id_embeddings[1:])

    # Attempt roc analysis on the following parameters
    FPR = 0
    recall = 0
    gthresholds = 0

    # Starting ROC analysis
    SLICE_BEGIN = 0
    SLICE_END = 1000
    IDENTITIES_TO_ROC_AGAINST = slice(SLICE_BEGIN, SLICE_END)
    SLICE_LENGTH = SLICE_END - SLICE_BEGIN
    max_auc = -1
    max_auc_id = 0

    i = 0

    # Do predictions only once
    predictions_impostor = model.predict(np.array(lfw_people.images_resized[IDENTITIES_TO_ROC_AGAINST]))

    # ROC curve
    #for i in range(len(testing_thresholds)):

    # Derive embeddings
    print("\n[ROC Analysis] Prediction ", i, " complete.")
    #print("\n[ROC Analysis] Running for threshold ", testing_thresholds[i])
    print("[ROC Analysis] Comparing with current picture which has the embedding ", test_emb)
    print(predictions_impostor)

    # Get results of cosine similarity with current face
    loss_impostor = keras.losses.cosine_similarity(
        predictions_impostor,
        test_emb
    )
    loss_genuine = keras.losses.cosine_similarity(
        rest_emb,
        test_emb
    )

    # Calculate using a short new session
    ilosspy = -loss_impostor.eval(session = tf.compat.v1.Session())
    glosspy = - loss_genuine.eval(session = tf.compat.v1.Session())
    print("   Cosine similarity for impostors ", ilosspy)
    print("   Cosine similarity for genuine ", glosspy)

    # Convert to matches vector for roc_curve
    imatches = [1 if x > testing_thresholds[i] else 0 for x in ilosspy]
    gmatches = [1 if x > testing_thresholds[i] else 0 for x in glosspy]
    imatches2 = [x for x in ilosspy]
    gmatches2 = [x for x in glosspy]
    print("Impostor similarity converted to {0,1} = ", imatches)
    print("Genuine similarity converted to {0,1} = ", gmatches)
    matches = np.append(imatches, gmatches)

    # try to do ROC on non-binary classification
    matches = np.append(imatches2, gmatches2)

    # Add the subsection of LFW and the current identity photos
    complete_y_true = np.append(np.zeros(SLICE_LENGTH), np.ones(NUMBER_OF_FACES - 1))
    complete_y_score = matches
    print("ROC: y_true = ", complete_y_true, ", y_score = ", complete_y_score)

    FPR, recall, gthresholds = sklearn.metrics.roc_curve(y_true = complete_y_true, y_score = complete_y_score)
    roc_auc = sklearn.metrics.auc(FPR, recall)
    if roc_auc > max_auc:
        max_auc_id = i
        max_auc = roc_auc
    print("FPR = ", FPR, ", recall = ", recall, ", thresholds = ", gthresholds)

    # Debug plot
    plt.plot(FPR, recall, colors[i], label='AUC' + str(i) + ' %s = %0.2f' % ('facenet', roc_auc))

    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(loc = 'lower right')
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.title('ROC Curve')
    plt.show()

    print("Best ROC AUC = ", max_auc, " of threshold ", testing_thresholds[max_auc_id])

    # find optimum threshold
    optimal_cutoff = Find_Optimal_Cutoff(complete_y_true, complete_y_score)
    print("Optimal threshold found at = ", optimal_cutoff)
    return optimal_cutoff
    #sys.exit(0)

    # Return index of best threshold
    #return max_auc_id


# Learn a new identity - register new face
def learning_menu():
    global state
    global counter
    global STEPS
    global face
    global identities
    global NUMBER_OF_FACES
    global testing_thresholds
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

    MAX_MESSAGES = NUMBER_OF_FACES - 1

    current_message = 0
    timer_tick = 0
    new_identity = []
    best_threshold = 0

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
            # Queue ROC analysis
            best_threshold = roc_analysis(new_identity)

            # Store embeddings
            new_identity = np.asarray(new_identity)
            new_identity = np.append(new_identity, best_threshold)#testing_thresholds[best_threshold])
            print("New identity ", new_identity)
            print("ROC analysis yielded best threshold as = ", best_threshold)#testing_thresholds[best_threshold])
            print("Previous identities = ", len(identities), " = ", identities)

            if len(identities) != 0:
                identities = np.insert(identities, identities.shape[0], new_identity, axis=0)
            else:
                identities = np.expand_dims(new_identity, axis=0)
            print("New identities = ", len(identities), " = ", identities)

            #new_identity.clear()
            new_identity = np.array([])
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
                # print("Embedding = ", embedding)

                # Append embedding to new identity
                new_identity.append(embedding)
                print("Embeddings size = ", len(new_identity))

                # Increment message and take new photo
                current_message = current_message + 1

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
