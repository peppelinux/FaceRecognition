This is a personal study about Face Recognition.
Started from a fork of https://github.com/shantnu/FaceDetect/

Requirements
-----------
It works with OpenCV3.

````
pip install opencv-python
````

Run
---
`python face_detect.py your_image.jpg`

`find ./ciao/VIDEOSORVEGLIANZA_STORAGE/INGRESSO_ARMADIO_ETH/2019-03*  -type f -name "*jpg" -exec python3 face_detect.py {} \;`

Better code and approaches
--------------------------

- https://github.com/ageitgey/face_recognition

Resources
---------
If you want to understand how the code works, the details are here:
https://realpython.com/blog/python/face-recognition-with-python/

Other:
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html
- https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
- https://www.learnopencv.com/invisibility-cloak-using-color-detection-and-segmentation-with-opencv/#:~:text=The%20Hue%20values%20are%20actually,detection%20of%20skin%20as%20red.
- https://stackoverflow.com/questions/42065405/remove-noise-from-threshold-image-opencv-python
