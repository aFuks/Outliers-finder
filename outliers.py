from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.cluster import KMeans
import os
from PIL import Image
import shutil
from sklearn.preprocessing import normalize
import cv2
import numpy as np
from sklearn.ensemble import IsolationForest
from shutil import copyfile
def extract_sift_features(image_path, max_keypoints=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Ograniczenie liczby punktów kluczowych
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_keypoints]
    descriptors = np.array([descriptor for keypoint, descriptor in zip(keypoints, descriptors)])

    return descriptors
def oblicz_liczbe_kolorow(image):
    # Przekształć obraz do tablicy NumPy
    np_image = np.array(image)

    # Zlicz różne kolory w obrazie
    unique_colors = np.unique(np_image.reshape(-1, np_image.shape[2]), axis=0)

    # Zwróć liczbę unikalnych kolorów
    return len(unique_colors)

def znajdz_zdjecie_z_najmniej_kolorow(folder_path, outliers_folder):
    najmniej_kolorow = float('inf')
    plik_z_najmniej_kolorow = None

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Sprawdź, czy to plik obrazu
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg')):
            try:
                # Spróbuj otworzyć obraz
                image = Image.open(file_path)

                # Oblicz liczbę kolorów w obrazie
                liczba_kolorow = oblicz_liczbe_kolorow(image)

                # Aktualizuj najmniej kolorów i nazwę pliku, jeśli obecny plik ma mniej kolorów
                if liczba_kolorow < najmniej_kolorow:
                    najmniej_kolorow = liczba_kolorow
                    plik_z_najmniej_kolorow = filename

                # Zamknij obraz
                image.close()
            except Exception as e:
                print(f"Błąd podczas przetwarzania pliku {filename}: {e}")
                continue  # Kontynuuj przetwarzanie kolejnego pliku w przypadku błędu

    if plik_z_najmniej_kolorow:
        # print(f"Najmniej kolorów ma plik: {plik_z_najmniej_kolorow} (Liczba kolorów: {najmniej_kolorow})")

        # Przenieś plik do folderu outliers
        outliers_path = os.path.join(outliers_folder, plik_z_najmniej_kolorow)
        shutil.copy(os.path.join(folder_path, plik_z_najmniej_kolorow), outliers_path)

def extract_resnet_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    features = np.squeeze(features)
    return features

input_folder = "./Final_images_dataset"
os.makedirs("./outliers_finished", exist_ok=True)

images_folder_path = input_folder

# Uzyskaj listę plików w folderze
image_files = [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]

# Pełne ścieżki do zdjęć
# image_paths = [os.path.join(images_folder_path, file) for file in image_files]

# Załaduj model ResNet-50
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

znajdz_zdjecie_z_najmniej_kolorow(input_folder, "outliers_finished")

image_features = []

def isolationforest(path, pathout, imagenumber, cont, ranst):
    folder_path = path
    outliers_folder = pathout

    # Utwórz folder na obrazy-odstępstwa, jeśli nie istnieje
    if not os.path.exists(outliers_folder):
        os.makedirs(outliers_folder)

    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith((".jpg", ".jpeg"))]

    # Wczytanie danych i ekstrakcja cech SIFT
    buf = [extract_sift_features(image_path) for image_path in image_paths]
    buf = np.array(buf)
    sift_features_list = buf.reshape(imagenumber, -1)

    # Przygotowanie danych uczących
    X_train = np.vstack(sift_features_list)

    # Standaryzacja danych
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)
    X_train = (X_train - mean_train) / std_train

    # Trenowanie modelu Isolation Forest
    isolation_forest_model = IsolationForest(contamination=cont, random_state=ranst)
    isolation_forest_model.fit(sift_features_list)

    # Otrzymaj predykcje od obu modeli
    predictions_isolation_forest = isolation_forest_model.predict(sift_features_list)

    # Ustal, ile modeli uznaje obraz za odstający
    num_models_identifying_outlier = (predictions_isolation_forest == -1)

    # Wybierz obrazy, które są uznawane za odstające przez co najmniej jeden model
    outliers = np.where(num_models_identifying_outlier > 0)[0]

    # Przenoszenie obrazów-odstępstw do folderu outliers_folder
    for outlier_index in outliers:
        if outlier_index < len(image_paths):  # Dodaj sprawdzenie, czy indeks mieści się w granicach listy
            outlier_path = image_paths[outlier_index]
            file_name = os.path.basename(outlier_path)
            new_path = os.path.join(outliers_folder, file_name)
            copyfile(outlier_path, new_path)

    return image_paths

# szukanie outlierów
image_paths = isolationforest('./Final_images_dataset', './outliers_not_finished', 104, 0.09, 42)
isolationforest('./outliers_not_finished','outliers_finished', 10, 0.24, 222)


for img_path in image_paths:
    features = extract_resnet_features(img_path)
    image_features.append(features)

image_features = np.array(image_features)

num_images, height, width, channels = image_features.shape
image_features_2d = image_features.reshape((num_images, height * width * channels))

# Apply K-means on the reshaped array
image_features_2d_normalized = normalize(image_features_2d, axis=1)
kmeans = KMeans(n_clusters=8, random_state=47)
kmeans.fit(image_features_2d_normalized)
labels = kmeans.labels_


# Twórz mapowanie między etykietami klastrów a kategoriami
cluster_category_mapping = {0: 'cluster_1', 1: 'cluster_2', 2: 'cluster_3', 3: 'cluster_4', 4: 'cluster_5',
                            5: 'cluster_6', 6: 'cluster_7', 7: 'cluster_8'}


# Przyporządkuj kategorie do zdjęć na podstawie etykiet klastrów
photo_category_mapping = {}
for i, img_path in enumerate(image_paths):
    category = cluster_category_mapping[labels[i]]
    photo_category_mapping[img_path] = category

# Utwórz folder do przechowywania kategorii
output_base_folder = "./data_categorised"
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

# Utwórz foldery dla każdej kategorii
for category in cluster_category_mapping.values():
    category_folder = os.path.join(output_base_folder, category)
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)

# Przenieś zdjęcia do odpowiednich folderów na podstawie przyporządkowanych kategorii
for img_path, category in photo_category_mapping.items():
    output_category_folder = os.path.join(output_base_folder, category)
    output_path = os.path.join(output_category_folder, os.path.basename(img_path))

    # Przenieś plik do nowego folderu
    shutil.copy(img_path, output_path)




