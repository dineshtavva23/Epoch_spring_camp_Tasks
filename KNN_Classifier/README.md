
# 🍎🍌🍊 KNN Classifier from Scratch (Using NumPy)

This is a simple Python project where we build a **K-Nearest Neighbors (KNN)** classifier using only **NumPy**. No external machine learning libraries are used.

---

## 📚 What is KNN?

**KNN** is a basic ML algorithm. It works like this:
1. When you give a new point (like a fruit), it looks at the **k nearest points** in the training data.
2. It checks which label (Apple, Banana, Orange) is the most common among those neighbors.
3. It assigns that label to the new point.

---

## 🧾 What’s in the Data?

Each row in the data looks like this:

```python
[weight, size, color_code, label]
```

- Example: `[150, 7.0, 1, 'Apple']`
- We convert labels (`'Apple'`, `'Banana'`, `'Orange'`) into numbers:
  - `'Apple' → 0`
  - `'Banana' → 1`
  - `'Orange' → 2`

---

## 🔧 What the Code Does

### 1. **Prepares the data**
- Encodes labels to numbers
- Separates features (`X`) and labels (`y`)

### 2. **Defines distance functions**
- **Euclidean**: straight-line distance
- **Manhattan**: like grid or city-block distance
- **Minkowski**: general form (we used `p=4`)

### 3. **Builds the KNN classifier**
- Finds the `k` closest neighbors for each test point
- Picks the most common label among those neighbors

### 4. **Predicts test data**
- Predicts labels for some new fruit examples

### 5. **Tests with different k values**
- Tries `k = 1`, `3`, and `5` to see how results change

### 6. **Accuracy function**
- Compares predicted vs actual labels

### 7. **Normalization (Min-Max Scaling)**
- Scales features between 0 and 1 to make things fair

---

## 🧪 Example Predictions

With test data like:

```python
[118, 6.2, 0]  → Banana  
[160, 7.3, 1]  → Apple  
[185, 7.7, 2]  → Orange
```

The model predicts:
```bash
Predictions: [1 0 2]
Converted Predictions: ['Banana', 'Apple', 'Orange']
```

It works great for all tested `k` values!

---

## 📈 Accuracy

We also test how accurate the model is by splitting the data:
- 75% for training
- 25% for testing

Then we calculate how many labels were predicted correctly.

---

## 🛠 How to Run

Make sure you have Python and NumPy installed. Then just run:

```bash
python knn_classifier.py
```

---

## ✅ Requirements

Just one library:
```bash
pip install numpy
```

---

## 📌 Future Ideas

- Add weighted KNN (closer neighbors get more vote)
- Add support for real-world datasets
- Use k-fold validation

---

## ✌️ Final Note

This project is perfect for beginners who want to **understand how KNN works** from the ground up!
