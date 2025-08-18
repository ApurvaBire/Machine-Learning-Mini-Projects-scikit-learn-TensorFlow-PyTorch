import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train_cat, epochs=5, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(x_test, y_test_cat)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# -----------------------------
# Visualization of prediction
# -----------------------------
# Pick a random test image
index = np.random.randint(0, x_test.shape[0])
sample_image = x_test[index]
true_label = y_test[index]

# Predict
pred_prob = model.predict(sample_image.reshape(1,28,28))
pred_label = np.argmax(pred_prob)

# Show image and prediction
plt.imshow(sample_image, cmap='gray')
plt.title(f"True: {true_label}\nPredicted: {pred_label}")
plt.axis("off")
plt.show()
