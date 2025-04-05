import numpy as np
import matplotlib.pyplot as plt
import cv2
gray_image = cv2.imread("gray_scale_image.jpeg",cv2.IMREAD_GRAYSCALE)

a1= gray_image
a2=np.fliplr(gray_image)
a3=np.flipud(gray_image)
a4=np.fliplr(np.flipud(gray_image))


upper_half = np.hstack([a1,a2])
lower_half = np.hstack([a3,a4])

output_image = np.vstack([upper_half,lower_half])

fig,(initial,final_output) = plt.subplots(1,2,figsize=(12,4))
initial.imshow(gray_image,cmap="gray")
initial.set_title("Original Image")
initial.axis("off")

final_output.imshow(output_image,cmap="gray")
final_output.set_title("Final Image after Kaleidoscope effect")
final_output.axis("off")

plt.tight_layout()
plt.show()