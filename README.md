This is a personal study about Face Recognition.
Started from a fork of https://github.com/shantnu/FaceDetect/

Requirements
-----------
It works with OpenCV3.

````
pip install python-opencv
````

Run
---
`python face_detect.py your_image.jpg`

`find ./ciao/VIDEOSORVEGLIANZA_STORAGE/INGRESSO_ARMADIO_ETH/2019-03*  -type f -name "*jpg" -exec python3 face_detect.py {} \;`

Resources
---------
If you want to understand how the code works, the details are here:
https://realpython.com/blog/python/face-recognition-with-python/
