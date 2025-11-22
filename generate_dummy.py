import matplotlib.pyplot as plt
import os

# path folder sekarang (folder tempat file ini berada)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# folder static/img di dalam webapp
img_folder = os.path.join(BASE_DIR, "static", "img")

# buat folder jika belum ada
os.makedirs(img_folder, exist_ok=True)

# buat grafik dummy
plt.figure()
plt.plot([1, 2, 3, 4], [10, 20, 15, 25], marker='o')
plt.title("Grafik Dummy Flood")

# simpan gambar ke static/img
output_path = os.path.join(img_folder, "dummy.png")
plt.savefig(output_path)
plt.close()

print(f"Gambar dummy dibuat di: {output_path}")
