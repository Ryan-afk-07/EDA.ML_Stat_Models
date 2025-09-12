import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from skimage import exposure, img_as_float
import numpy as np

# Image parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30  # Increase if you want better accuracy

def AHE(img):
    # normalize to [0,1] float
    img = img / 255.0 if img.max() > 1.0 else img  
    
    # apply CLAHE channel-wise
    img_out = np.zeros_like(img)
    for i in range(img.shape[-1]):
        img_out[..., i] = exposure.equalize_adapthist(img[..., i], clip_limit=0.03)
    return img_out

# Data augmentation & normalization
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    )

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


val_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

x, y = next(train_generator)
print("Batch shape:", x.shape, y.shape)
print("Pixel range:", x.min(), x.max())

# Save class indices for later use in prediction
import json
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# Model architecture
model = Sequential([
    Conv2D(32, (5, 5), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    #Flatten()
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Do callbacks
lr_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Save best model
checkpoint = ModelCheckpoint('eye_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, lr_reduction]
)

print("Training complete. Model saved as eye_model.h5")
