Image Pre-Processing
====================
Documentation of ricebowl image preprocessing.

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
Resizes an image

Parameters- Image, Length, Width

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



