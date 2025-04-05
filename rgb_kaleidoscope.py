import numpy as np 
import matplotlib.pyplot as plt
import cv2
rgb_image= cv2.imread("rgb_image.jpeg")

rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

a1= rgb_image
a2=np.fliplr(rgb_image)
a3=np.flipud(rgb_image)
a4=np.fliplr(np.flipud(rgb_image))

upper_half = np.hstack([a1,a2])
lower_half = np.hstack([a3,a4])

output_image = np.vstack([upper_half,lower_half])
fig,(initial,final_output) = plt.subplots(1,2,figsize=(12,4))
initial.imshow(rgb_image)
initial.set_title("Original Image")
initial.axis("off")

final_output.imshow(output_image)
final_output.set_title("Final Image after Kaleidoscope effect")
final_output.axis("off")


plt.tight_layout()
plt.show()