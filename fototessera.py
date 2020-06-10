import cv2 as cv
import numpy as np
import sys

def filter_dimensions(image, faces):
    # check that the image is not too much greater than the detected face
    image_rap = image.shape[:2]
    face = faces[0]
    # it's a rectangle!
    face_rap = face[3], face[3]

    rap_x = '{:01.3f}'.format( face_rap[0] / image_rap[0])
    rap_y = '{:01.3f}'.format( face_rap[1] / image_rap[1])
    min_rapp = 0.3
    if float(rap_x) < min_rapp or float(rap_x) < min_rapp:
        _msg = ("The Face is too small ({}) in relation to the size of "
                "the image {}. This must be at least {}").format(face_rap,
                                                                 image_rap,
                                                                 min_rapp)
        raise Exception(_msg)
        return False
    return True


def filter_n_faces(faces, n=1):
    n_faces = len(faces)
    if not n_faces:
        raise Exception("No face detected")
        return False
    if n_faces > n:
        raise Exception("Found more than 1 faces, invalid.")
        return False
    return True


def face_detection(casc, image):
    # Create the haar cascade
    faceCascade = cv.CascadeClassifier(casc)

    # Read the image
    #  image = cv.imread(imagePath)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.01,
        minNeighbors=5,
        # the size of the returning rectangle
        minSize=(30, 30),
    )
    msg = "Found {0} faces [{1}]".format(len(faces),
                                             casc)
    print(msg)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        img = cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    try:
        filter_n_faces(faces)
        filter_dimensions(image, faces)
    except Exception as e:
        print(e)
        return 0, image

    return cv, image


def resize(image):
    # resize
    dst_width = 300
    rap = (dst_width / image.shape[1])
    dim = (int(rap * image.shape[1]), int(rap * image.shape[0]))
    print('Dimensions converted from ', image.shape[:2], 'to ', dim)
    resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    image = resized
    return image


def is_fototessera(image, debug=False):
    #  image = cv.imread(cv.samples.findFile(image_path))

    # Convert BGR to HSV - Hue Saturation Brightness
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # filter by mean
    gray = cv.equalizeHist(gray)
    _, gray = cv.threshold(gray,127,255,cv.THRESH_TRUNC)
    gray = cv.equalizeHist(gray)

    middle_top = int(gray.shape[0]/2)
    left_border = int(gray.shape[1]/10)
    right_border = gray.shape[1] - int(gray.shape[1]/10)

    crop_left = gray[0:middle_top, 0:left_border]
    crop_right = gray[0:middle_top, right_border:gray.shape[1]]

    if crop_left.mean() > 250 and crop_right.mean() > 250:
        return 1, image
    else:
        print('Noisy Background, invalidate.')
        if debug:
            # [y1:y2, x1:x2]
            cv.imshow('crop', gray[0:middle_top, 0:gray.shape[1]])
            #  cv.imshow('crop_left', crop_left)
            #  cv.imshow('crop_right', crop_right)
        return 0, image


def repr_invalid(image, title='invalid'):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    g2rgb = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    hsv = cv.cvtColor(g2rgb, cv.COLOR_BGR2HSV)
    cv.imshow(title,hsv)
    cv.waitKey(0)
    cv.destroyAllWindows()
    sys.exit(1)


if __name__ == '__main__':
    image_path = sys.argv[1]
    debug = 1

    print('Processing: {} ...'.format(image_path))
    image = cv.imread(cv.samples.findFile(image_path))
    image = resize(image)
    res, image = is_fototessera(image, debug)

    if debug:
        if res:
            cv.imshow('fototessera', image)
        else:
            repr_invalid(image)

    #  casc = 'haarcascades/haarcascade_frontalface_alt.xml'
    casc = 'haarcascades/haarcascade_frontalface_alt2.xml'
    res, image = face_detection(casc,
                                image)
    if debug:
        if res:
            cv.imshow('fototessera', image)
        else:
            repr_invalid(image)

    cv.waitKey(0)
    cv.destroyAllWindows()

# for i in `ls examples/fototessere/`; do  python fototessera.py  examples/fototessere/$i; done
