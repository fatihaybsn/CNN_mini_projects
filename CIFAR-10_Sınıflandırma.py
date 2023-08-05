import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report


# CIFAR-10 sınıf isimleri
CLASS_LABELS = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]


def build_model(input_shape, num_classes=10):
    """Basit CNN mimarisi (CIFAR-10 için)."""
    model = Sequential()

    # Feature Extraction Block 1
    model.add(
        Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
            input_shape=input_shape,
        )
    )
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Feature Extraction Block 2
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))  # 62 saçmalığı düzeltildi
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classification Head
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def main():
    # 1) Veri setini yükle
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Örnek birkaç görüntü göster (opsiyonel, ama hoş duruyor)
    fig, axes = plt.subplots(1, 5, figsize=(12, 4))
    for i in range(5):
        axes[i].imshow(x_train[i])
        label = CLASS_LABELS[int(y_train[i])]
        axes[i].set_title(label)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

    # 2) Normalizasyon
    # Hem eğitim hem test verisini [0, 1] aralığına getir
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 3) One-hot encoding
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # 4) Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    datagen.fit(x_train)

    # 5) Modeli oluştur ve derle
    model = build_model(input_shape=x_train.shape[1:], num_classes=num_classes)

    model.compile(
        optimizer=RMSprop(learning_rate=1e-4, decay=1e-6),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # 6) Eğit
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=30,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    # 7) Test seti üzerinde değerlendirme
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

    # 8) Classification report
    y_pred = model.predict(x_test, batch_size=64, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred_classes, target_names=CLASS_LABELS))

    # 9) Eğitim süreci grafikleri
    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 10) Modeli kaydet
    model.save("cifar10_cnn.keras")
    print("\nModel `cifar10_cnn.keras` olarak kaydedildi.")


if __name__ == "__main__":
    main()
