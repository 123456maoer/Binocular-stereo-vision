Tutorial for the dual-target calibration code set:
1. gain.py collects data
2. cut.py crops images
3. calib.py performs calibration, generates a YAML file, and generates calibrated images. 

4. splice.py combines the rectified stereo images into a single image.
5. jpg2avi.py stitches the stereo images into a video in AVI format.
6. calibSGBM.py calculates 3D coordinates and generates a color depth map. 

4.1 If you only want to correct a single image, modify the calibSGBM.py file, or use calibSGBM2py.
