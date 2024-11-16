import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os

# Load the mel-spectrogram from the .npy file
mel_spectrogram = np.load('/home/jeonyj0612/SpeechDis/preprocessed_data/LibriTTS-100/mel/19-mel-19_198_000000_000000.npy')  # Replace with your file path
num_mel_bins, time_steps = mel_spectrogram.shape

# Convert the numpy array to a torch tensor for masking
spec = torch.tensor(mel_spectrogram.copy())

# Masking parameters
R = 10  # Maximum time mask parameter
F = 10  # Maximum frequency mask parameter
mR = 0  # Number of frequency masks
mF = 4  # Number of time masks

# Define the masking functions as previously outlined
def time_masking(spec, T, R, mR):
    for _ in range(mR):
        tau = random.randint(0, R)
        t = random.randint(0, max(0, T - tau))
        spec[:, t:t + tau] = 0
    return spec

def frequency_masking(spec, num_mel_bins, F, mF):
    for _ in range(mF):
        phi = random.randint(0, F)
        f = random.randint(0, max(0, num_mel_bins - phi))
        spec[f:f + phi, :] = 0
    return spec

# Apply both time and frequency masking
masked_spec = time_masking(spec.clone(), time_steps, R, mR)
masked_spec = frequency_masking(masked_spec, num_mel_bins, F, mF)

# Convert the masked torch tensor back to numpy for visualization
masked_spec = masked_spec.numpy()
print(masked_spec.shape)

# Directory to save images
# output_dir = 'output_mel_images'
# os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Plot and save the images
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Original mel-spectrogram
axs[0].imshow(mel_spectrogram.T, aspect='auto', origin='lower', cmap='viridis')
axs[0].set_title('Original Mel-Spectrogram')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Frequency')

# Masked mel-spectrogram
axs[1].imshow(masked_spec.T, aspect='auto', origin='lower', cmap='viridis')
axs[1].set_title('Masked Mel-Spectrogram')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Frequency')

# Save the figure to the specified folder
output_path = os.path.join('mel_spectrogram_comparison.png')
plt.tight_layout()
plt.savefig("/home/jeonyj0612/SpeechDis/test")
plt.close(fig)  # Close the figure after saving

print(f"Mel-spectrogram images saved to {output_path}")
