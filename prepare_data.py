import os
import cv2
import numpy as np

data_folder = "C:\\Users\\Valentin\\Documents\\HTL\\4cHel\\KI_SYS\\Supervised_Learning\\Neuronales_Netz\\Buchstaben_Recognition\\BigDataSet"
final_size = 28     # so groß sind die Bilder z.B. 28x28

letters = []
labels = []

# alle Buchstaben durchgehen
for label in os.listdir(data_folder):
    label_folder = os.path.join(data_folder, label)                 # Pfad-aktualisieren

    # alle .png's durchgehen
    for file in os.listdir(label_folder):
        file_path = os.path.join(label_folder, file)                # Pfad-aktualisieren
            
        image = cv2.imread(file_path, 0)                            # Graustufe umwandeln

        smaller_img = cv2.resize(image, (final_size, final_size))   # auf 28x28 skalieren
        final_img = 1 - (smaller_img / 255)                         # von 0-255 => 0-1

        letters.append(final_img)
        labels.append(ord(label.upper()) - ord('A'))                # ord() gibt den ASCII Zeichenwert eines Chars zurück 
                                                                    # so ist z.B. ord(A) - ord(A) = 65 - 65 = 0
# Erzeugt NxRxC Arrays                                                so ist z.B. ord(C) - ord(C) = 67 - 65 = 2
X_data = np.array(letters)
y_data = np.array(labels).reshape(-1, 1)
# -1 erster Parameter => Dimensionen des Arrays                     (-1 => automatisch berechnen) ODER Platzhalter
# 1 zweiter Parameter => Anzahl an Elementen pro Zeile/Spalte       (-1 => automatisch berechnen)

np.save("X_data.npy", X_data)
np.save("y_data.npy", y_data)
