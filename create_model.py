import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, MaxPooling2D, Dropout, SeparableConv2D
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import numpy as np
from sklearn.model_selection import train_test_split

# Ladet numpy Array Datensatz
x_data_letters = np.load("X_data.npy")
y_data_labels = np.load("y_data.npy")

# Zeigen von einem Buchstaben(nur zur Kontrolle)
"""
plt.imshow(x_data_letters[3], cmap='gray')  # Verwenden von 'gray' für Graustufenbilder
plt.axis('off')  # Entfernt die Achsen
plt.show()
"""
# Konfigurieren von der zufälligen "Bildverzerrung" => Meine Werte
datagen = ImageDataGenerator(
    rotation_range=15,       # Random Rotation
    width_shift_range=0.1,   # Random Breite
    height_shift_range=0.1,  # Random Höhe
    shear_range=0.2,
    zoom_range=0.2,          # Zoom-In und Zoom-Out um bis zu 20%
    horizontal_flip=False,   # Horizontales Spiegeln (macht bei Buchstaben keinen Sinn)
    fill_mode="nearest"      # Fehlende Pixel durch nächste Werte ersetzen
)

x_data_letters_train, x_data_letters_test, y_data_labels_train, y_data_labels_test = train_test_split(
    x_data_letters, y_data_labels, test_size=0.1, train_size=0.9, random_state=100)

# Extra Dimension für ImageDataGenerator(1 für Grayscale)
x_data_letters_train = x_data_letters_train.reshape(-1, 28, 28, 1)
# Verzerrer auf Daten anwenden, nur unbedingt Notwendig für statistische Sachen
datagen.fit(x_data_letters_train)
# erstellt Generator(durch den laufen die Bilder dann in der Batchsize durch das Modell), kann auch simple Modifikationen anwenden dann braucht man fit nicht
# Batchsize: Wie viele Daten das Modell bekommt bevor die Gewichte angepasst wird. größere Batchsize => längere Dauer
generator = datagen.flow(x_data_letters_train, y_data_labels_train, batch_size=32)

# Sequentiell = hidden layers nacheinander
# Modell initialisieren
model = Sequential([
    # Erste Schicht
    SeparableConv2D(filters=64, kernel_size=(3, 3), activation='leaky_relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    
    # Zweite Schicht
    SeparableConv2D(filters=128, kernel_size=(3, 3), activation='leaky_relu', depthwise_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    
    # Dritte Schicht
    SeparableConv2D(filters=256, kernel_size=(3, 3), activation='leaky_relu', depthwise_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(rate=0.3),
    Flatten(),
    
    # Vierte Schicht
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(rate=0.5),
    
    # Fuenfte Schicht
    Dense(64, activation='relu'),   
    Dense(26, activation='softmax'),
])

# Compile the Model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Fit the model
model.fit(generator, epochs=100, validation_data=(x_data_letters_test, y_data_labels_test))

# Print the Loss
test_loss, test_acc = model.evaluate(x_data_letters_test, y_data_labels_test, verbose=2)
print(f"\nTest Accuracy: {test_acc}")

# Save the model
model.save("my_tf_model.h5")
print(f"Model saved as my_tf_model.h5")