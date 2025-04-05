These are the solutions for Task-1 of Epoch Spring Camp.
Explanation:
1.Gray Scale Kaleidoscope:
                          I downloaded a grayscale image named gray_scale_image.jpeg and loaded it into a NumPy array using the cv2 library. From this base image, I created three transformed copies: a2 by flipping the image left-to-right, a3 by flipping it top-to-bottom, and a4 by rotating the original image 90 degrees.
The original image (a1) and the horizontally flipped image (a2) were combined using np.hstack() to form the upper half of the final kaleidoscope. Similarly, the vertically flipped (a3) and rotated (a4) versions were stacked to form the lower half. Finally, I used np.vstack() to combine both halves vertically and displayed the resulting kaleidoscope effect using matplotlib.
2.RGB Kelidoscope: 
                   I used same logic as used in creating kaleidoscope for grayscale image but while loading a RGB image using cv2 library it loads images in BGR format , so to convert it back into RGB foramt we can use cv2.cvtColor(image,cv2.COLOR_BGR2RGB). This step ensures that the colors are displayed correctly when using Matplotlib. The image was then transformed and combined in the same way as the grayscale version to distplay the final RGB kaleidoscope effect.

