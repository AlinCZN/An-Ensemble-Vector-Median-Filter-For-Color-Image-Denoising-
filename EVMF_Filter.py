import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util

def filter_image_with_sbvf(noisy_img, window_size=3):
    half_window = window_size // 2
    rows, cols, _ = noisy_img.shape
    img_filtered = noisy_img.copy()

    for i in range(half_window, rows - half_window):
        for j in range(half_window, cols - half_window):
     
            window = noisy_img[i-half_window:i+half_window+1, j-half_window:j+half_window+1, :]
            center_pixel = noisy_img[i, j, :].reshape(1, 3)
            neighbors = window.reshape(-1, 3)

            distances = np.linalg.norm(neighbors - center_pixel, axis=1)

            weights = np.exp(-distances**2)
            weights /= weights.sum()

            weighted_sum = np.sum(neighbors * weights[:, np.newaxis], axis=0)
            img_filtered[i, j, :] = weighted_sum

    return img_filtered
def filter_image_with_avmf(noisy_img, window_size=3, threshold=0.2):
    rows, cols, _ = noisy_img.shape
    img_filtered = noisy_img.copy()
    half_window = window_size // 2

    for r in range(half_window, rows - half_window):
        for c in range(half_window, cols - half_window):
            window = noisy_img[r-half_window:r+half_window+1, c-half_window:c+half_window+1, :]
            neighbors = window.reshape(-1, 3)

            vector_median = np.median(neighbors, axis=0)

            central_pixel = noisy_img[r, c]
            central_distance = np.linalg.norm(central_pixel - vector_median)

            if central_distance > threshold:
                img_filtered[r, c, :] = vector_median
            else:
                img_filtered[r, c, :] = central_pixel

    return img_filtered

def filter_image_with_fpgf(noisy_img, window_size=3, distance_threshold=0.1):
    half_window = window_size // 2
    rows, cols, channels = noisy_img.shape
    filtered_img = np.zeros_like(noisy_img)

    for i in range(half_window, rows - half_window):
        for j in range(half_window, cols - half_window):

            window = noisy_img[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1, :]
            neighbors = window.reshape(-1, 3)  
            
            center_pixel = noisy_img[i, j, :].reshape(1, 3)
            distances = np.linalg.norm(neighbors - center_pixel, axis=1)
            peer_group = neighbors[distances < distance_threshold]
          
            if len(peer_group) > 3:
                
                red_mean = np.mean(peer_group[:, 0])
                green_mean = np.mean(peer_group[:, 1])
                blue_mean = np.mean(peer_group[:, 2])
                
                filtered_img[i, j, :] = [red_mean, green_mean, blue_mean]
            else:
                
                filtered_img[i, j, :] = np.median(neighbors, axis=0)

    return filtered_img
    
def filter_image_with_vmf(img, window_size=3):
    rows, cols, _ = img.shape
    filtered_img = img.copy() 

    for i in range(rows):
        for j in range(cols):
            half_window = window_size // 2
            row_min = max(i - half_window, 0)
            row_max = min(i + half_window + 1, rows)
            col_min = max(j - half_window, 0)
            col_max = min(j + half_window + 1, cols)

            neighbors = img[row_min:row_max, col_min:col_max, :]
            neighbors = neighbors.reshape(-1, 3) 

            vector_median = np.median(neighbors, axis=0) 
            filtered_img[i, j, :] = vector_median

    return filtered_img

img = io.imread("Penguins.jpg")
plt.figure(), plt.imshow(img), plt.title("Imagine originală"), plt.axis('off'), plt.show()
img = img[:, :, 0:3] / 255 
noisy_img = util.random_noise(img, mode='s&p', amount=0.1)
plt.figure(), plt.imshow(noisy_img), plt.title("Imagine cu zgomot"), plt.axis('off'), plt.show()

img_sbvf = filter_image_with_sbvf(noisy_img)
plt.figure(), plt.imshow(img_sbvf), plt.title("Imagine Filtrată SBVF"), plt.axis('off'), plt.show()
img_sbvf = (img_sbvf * 255).astype(np.uint8)

img_avmf = filter_image_with_avmf(noisy_img)
img_avmf = (img_avmf * 255).astype(np.uint8)
plt.figure(), plt.imshow(img_avmf), plt.title("Imagine Filtrată AVMF"), plt.axis('off'), plt.show()

img_fpgf = filter_image_with_fpgf(noisy_img)
plt.figure(), plt.imshow(img_fpgf), plt.title("Imagine Filtrată FPGF"), plt.axis('off'), plt.show()
img_fpgf = (img_fpgf * 255).astype(np.uint8)

img_vmf = filter_image_with_vmf(noisy_img)
plt.figure(), plt.imshow(img_vmf), plt.title("Imagine Filtrată VMF"), plt.axis('off'), plt.show()
img_vmf = (img_vmf * 255).astype(np.uint8)

rows, cols, channels = img_avmf.shape
filtered_img = np.zeros_like(img_avmf)

def euclidean_norm(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

for i in range(rows):
    for j in range(cols):
        pixel_values = np.array([img_avmf[i, j], img_fpgf[i, j], img_sbvf[i, j], img_vmf[i, j]])
        distances = []
        for k in range(4):
            for l in range(k + 1, 4):
                distances.append(euclidean_norm(pixel_values[k], pixel_values[l]))
        
        min_dist_index = np.argmin(distances)
        min_dist_pixel = pixel_values[min_dist_index//3]  
        
        filtered_img[i, j] = min_dist_pixel
        

plt.figure()
plt.imshow(filtered_img)
plt.title("Imagine Finală Filtrată EVMF")
plt.axis('off')
plt.show()