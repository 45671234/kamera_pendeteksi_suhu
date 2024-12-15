import cv2
import numpy as np

# Fungsi untuk mendeteksi wajah dan menampilkan suhu tubuh
def deteksi_wajah_dan_suhu(video_source=0):
    # Load model deteksi wajah dari OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Buka kamera di Laptop
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Kamera tidak dapat dibuka!")
        return
    
    # Suhu tubuh sebelumnya (misal)
    suhu_sebelumnya = 32.0

    while True:
        # Baca frame dari video
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat membaca frame dari kamera!")
            break

        # Ubah ke grayscale untuk deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah di dalam frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
         # Perbarui suhu tubuh secara perlahan
        suhu_tujuan = round(np.random.uniform(31.0, 42.0), 1)
        suhu_sebelumnya += (suhu_tujuan - suhu_sebelumnya) * 0.05
        suhu_tubuh = round(suhu_sebelumnya, 1)

        # Suhu tubuh setiap wajah yang terdeteksi 
        for (x, y, w, h) in faces:
            # Membuat kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Membuat suhu tubuh antara 31.0 - 42.0
            suhu_tubuh = round(np.random.uniform(31.0, 42.0), 1)

            # Menampilkan suhu di atas kotak wajah
            cv2.putText(frame, f"Suhu : {suhu_tubuh}°C", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Tampilan frame dengan deteksi wajah dan suhu
        cv2.imshow('Deteksi Suhu Tubuh', frame)

        # Keluar jika tombol 'o' di tekan
        if cv2.waitKey(1) & 0xFF == ord('o'):
            break

    # Melepaskan resource
    cap.release()
    cv2.destroyAllWindows()

# Memanggil fungsi utama untuk memulai deteksi
if __name__ == "__main__":
    deteksi_wajah_dan_suhu()
