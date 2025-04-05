# Epoch Spring Camp - Task 1: Kaleidoscope Image Transformations

## Overview

This task involves creating a **kaleidoscope effect** using image processing techniques with Python and OpenCV. Two versions of the effect were implemented:

1. **Grayscale Kaleidoscope**
2. **RGB Kaleidoscope**

---

## 1. Grayscale Kaleidoscope

### Explanation:

- A grayscale image (`gray_scale_image.jpeg`) was loaded using the `cv2` library and converted into a NumPy array.
- From the base image (`a1`), three transformed versions were created:
  - `a2`: Horizontally flipped (`cv2.flip(a1, 1)`)
  - `a3`: Vertically flipped (`cv2.flip(a1, 0)`)
  - `a4`: Rotated 90 degrees (`cv2.rotate(a1, cv2.ROTATE_90_CLOCKWISE)`)
- The upper half of the kaleidoscope was formed by horizontally stacking `a1` and `a2` using `np.hstack()`.
- The lower half was formed by stacking `a3` and `a4` in the same way.
- Finally, the two halves were combined vertically using `np.vstack()` to create the final kaleidoscope image.
- The result was displayed using `matplotlib.pyplot`.

---

## 2. RGB Kaleidoscope

### Explanation:

- The same logic used for the grayscale version was applied to an RGB image.
- **Note:** When reading an image with OpenCV, it is loaded in **BGR format**. To convert it to RGB (so that colors display correctly with `matplotlib`), the following conversion was used:

  ```python
  image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
