import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def analyze_multichannel_npy(file_path):
    # Load data - expected shape (11, 128, 128) or (128, 128, 11)
    data = np.load(file_path).astype(np.float32)

    # Standardize shape to (channels, height, width)
    if data.shape[0] != 11:
        data = np.moveaxis(data, -1, 0)

    num_channels = data.shape[0]

    # --- VIEW 1: The Trellis (All Channels) ---
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    fig.suptitle(f"Channel Breakdown: {file_path}", fontsize=16)
    axes = axes.flatten()

    for i in range(num_channels):
        im = axes[i].imshow(data[i], cmap='magma')
        axes[i].set_title(f"Channel {i}")
        axes[i].axis('off')

    # Hide the 12th empty subplot
    axes[-1].axis('off')
    plt.tight_layout()
    plt.show()

    # --- VIEW 2: PCA (Dimensionality Reduction) ---
    # Reshape for PCA: (128*128, 11)
    flat_data = data.reshape(num_channels, -1).T
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(flat_data)

    # Reshape back to image format
    pca_img = pca_result.reshape(128, 128, 3)

    # Normalize PCA result to 0-1 for display
    pca_img = (pca_img - pca_img.min()) / (pca_img.max() - pca_img.min())

    plt.figure(figsize=(8, 8))
    plt.imshow(pca_img)
    plt.title("PCA Composite (Top 3 Variance Components as RGB)")
    plt.axis('off')
    plt.show()

    print(f"Variance explained by top 3 components: {np.sum(pca.explained_variance_ratio_):.2%}")

if __name__ == "__main__":
    input_dir = r'C:\Users\dcost\Chandra Mentorship\Example Dataset\input\Aug_29_2017_73.75.npy'
    analyze_multichannel_npy(input_dir)