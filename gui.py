import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class HandwritingRecognizerApp:
    def __init__(self, root):
        self.model = tf.keras.models.load_model("model_kisy.h5")

        self.root = root
        self.root.title("Buchstabenerkennung-App")

        self.canvas_width = 280
        self.canvas_height = 280

        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.grid(row=0, column=0)

        # internes PIL Image
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw_image = ImageDraw.Draw(self.image)

        self.pen_color = "black"
        self.brush_size = 8

        self.drawing = False
        self.last_x, self.last_y = None, None

        self.erase_button = tk.Button(root, text="Löschen", command=self.erase)
        self.erase_button.grid(row=1, column=0)

        self.predict_button = tk.Button(root, text="Auswerten", command=self.predict)
        self.predict_button.grid(row=2, column=0)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # erstellt die Figur/Graphen, subplots damit man mehr Zugriff auf einzelne Elemente hat
        self.figure, self.ax = plt.subplots(figsize=(6, 2.5))
        # "verpackt" den Plot als TK-Fenster => kann wie ein Button eingebaut werden
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=root)
        # platziert das Fenster
        self.canvas_plot.get_tk_widget().grid(row=0, column=1)

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            # einmal in TKinter zeichnen und einmal in PIL-Image(bischen andere Syntax für beide)
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill=self.pen_color, width=self.brush_size,
                                    capstyle=tk.ROUND, smooth=True)
            self.draw_image.line([self.last_x, self.last_y, x, y], fill=self.pen_color, width=self.brush_size)

            self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.drawing = False

    def erase(self):
        self.canvas.delete("all")
        self.image.paste(255, [0, 0, self.canvas_width, self.canvas_height])    # Löschen von PIL Image

    def predict(self):
        # dieselbe Logik wei bei prepare_data
        img = self.image.copy()
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = self.model.predict(img_array)
        probabilities = prediction[0]   # ins richtige Format für Matplotlib bringen
        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                  "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                  "U", "V", "W", "X", "Y", "Z"]
        
        # printen vom wahrscheinlichsten Buchstaben
        predicted_class = np.argmax(probabilities)
        predicted_letter = labels[predicted_class]
        print(predicted_letter)

        # aktualisieren von Plot
        self.ax.clear()
        self.ax.bar(labels, probabilities, color="skyblue")
        self.ax.set_ylim([0, 1])
        self.ax.set_title("Wahrscheinlichkeiten")
        self.ax.set_ylabel("Sicherheit")
        self.canvas_plot.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingRecognizerApp(root)
    root.mainloop()