import numpy as np
np.set_printoptions(linewidth=300, precision=3) # Tam matris gösterimleri için
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

x = np.load("distribution.npy")


# Grafikleri başlatır.
plt.ion()
figure, ax = plt.subplots(figsize=(15, 15))
ax.yaxis.set_inverted(True)


rotated_x = rotate(x,270)
# guncellenen tasarım alanını tekrar çizer.
ax.contourf(rotated_x, cmap="jet")
figure.canvas.draw()

# Grafiği sürekli açık tutar.
figure.canvas.flush_events()

plt.savefig("topoloji2.pdf")
plt.show()
