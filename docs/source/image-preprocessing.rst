Image Pre-Processing
====================
Documentation of ricebowl image preprocessing.
To use this simply do from ricebowl.processing import data_preproc and then use each function with image_preproc.<function>

read_image
^^^^^^^^^^
Reads a single image from the path.

Parameters- Path of the image

Output- Image

Usage::

    img=read_img('path of the image')


show_image
^^^^^^^^^^
Displays the image with a customized title.

Parameters- Image, title(optional; Default-'img')

Output- Image (Press any key to continue/exit)

Usage::

    img=show_image(img,title='my_title')



write_image
^^^^^^^^^^^
Writes the image at a given path location with the customized name.

Parameters- Path+<img_name.ext>, Image

Output- Image at the given location saved. (Please note: Checks for <if exists> have not been added. Same name will cause conflict)

Usage::

    img=write_image('./foldername/image.png', img)

 
inverting
^^^^^^^^^
Inverts an image (Converts to negative)

Parameters- Image

Output- Inverted Image

Usage::

    img=inverting(img)


gray_scale
^^^^^^^^^^
Converts an image to grayscale

Parameters- Image

Output- Gray-Scale Image

Usage::

    img=gray_scale(img)
 


resize
^^^^^^
Resizes an image keeping the aspect ratio constant

Parameters- Image, Width, Height

Output- Resized Image

Usage::

    img=resize(img, 100, 300)



gaussian_blurring
^^^^^^^^^^^^^^^^^
Blurs an image using gaussian blurring.

Parameters- Image, Kernel(Optional; Default- (21,21)

Output- Blurred Image

Usage::

    img=gaussian_blurring(img, ksize=(21,21))



orb_features
^^^^^^^^^^^^
Extracts features in the passed image

Parameters- Image

Output- Feature extracted image, Descriptor Matrix

Usage::

    img,desc=orb_features(img)



get_images
^^^^^^^^^^
Returns the array data of images and their labels (entire path)

Parameters- Path of the folder containing all images

Output- Data Matrix, Label array

Usage::

    data,labels=get_images('./path')



dcm_to_png
^^^^^^^^^^
Converts .dcm images to .png format (from the entire folder to another folder)

Parameters- Path of folder with input images in .dcm format, Path of folder to write the .png format

Output- Images written in the output folder. (Added print command showing how many images are written.)

Usage::

    dcm_to_png('./input_folder_path/','./output_folder_path/')

 

denoise
^^^^^^^
Removing noise from a colored Image

Parameters- Input Image (colored)

Output- Noise removed image

Usage::

    img=denoise(img)



binarization
^^^^^^^^^^^^
Converting an image to binary using Otsu Thresholding

Parameters- Input Image

Output- Black and white image

Usage::

    img=binarization(img)



erode
^^^^^
General function to erode text in Image (Performs morphological transformation - erosion)

Parameters- Input Image

Output- Eroded image

Usage::

    img=erode(img)



find_contours
^^^^^^^^^^^^^
Finding contours of an image

Parameters- Input Image

Output- Array of image contours

Usage::

    img=find_contours(img)



sharpen
^^^^^^^
Sharpens an image

Parameters- Input Image

Output- Sharpened image

Usage::

    img=sharpen(img)



edging
^^^^^^
Finding edges of an image (canny)

Parameters- Input Image

Output- Edges are outlined in the image

Usage::

    img=edging(img)



autorotate
^^^^^^^^^^
Auto rotating an image according to the degree of skewness identified

Parameters- Input Image

Output- De-skewed image

Usage::

    img=autorotate(img)



image_enhancer
^^^^^^^^^^^^^^
General function to adjust the image's contrast

Parameters- Input Image

Output- Image with increased contrast

Usage::

    img=image_enhancer(img)



remove_shadow
^^^^^^^^^^^^^
General function to remove shadows from image

Parameters- Input Image

Output- Image with shadows removed

Usage::

    img=remove_shadow(img)
