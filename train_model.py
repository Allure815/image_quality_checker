# train_model.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1️⃣ Data augmentation for small dataset
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # 20% validation
)

# 2️⃣ Load training and validation data
train_data = datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=4,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=4,
    class_mode='binary',
    subset='validation'
)

# 3️⃣ Build a small CNN
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary output
])

# 4️⃣ Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5️⃣ Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

# 6️⃣ Save the trained model
model.save("model/image_quality_model.h5")
print("✅ Model trained and saved as 'model/image_quality_model.h5'")
