# Wavelet-Based Image Denoising with Gradio Interface

This project provides an interactive web app to visualize the addition of Gaussian noise to an image and then denoise it using **Wavelet Transform** techniques. The user can control noise parameters and the wavelet settings through a simple Gradio-based UI.

## Features

- Add **Gaussian Noise** to an uploaded image.
- Perform **Wavelet Denoising** with adjustable:
  - Wavelet type (e.g., `db1` to `db6`)
  - Decomposition levels (1 to 5)
- View and download the **Noisy** and **Denoised** images.
- Clear/reset the inputs easily.
- Shareable public Gradio link.

## How It Works

1. **Noise Addition**:\
   Adds Gaussian noise with user-specified sigma value.

2. **Wavelet Decomposition**:\
   Decomposes the image channels separately into wavelet coefficients.

3. **Thresholding**:\
   Applies soft-thresholding to the detail coefficients based on estimated noise.

4. **Reconstruction**:\
   Reconstructs the denoised image from modified wavelet coefficients.

## Installation

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. Install the required libraries:

   ```bash
   pip install numpy opencv-python pywavelets gradio matplotlib
   ```

## Usage

Run the app:

```bash
python app.py
```

(Replace `app.py` with your actual file name if different.)

It will open a Gradio interface where you can upload an image, tune parameters, and visualize the results.

## Interface Description

| Control                 | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| **Upload Image**        | Load an input image for processing             |
| **Sigma Slider**        | Set standard deviation of added Gaussian noise |
| **Wavelet Type**        | Choose wavelet basis function for denoising    |
| **Decomposition Level** | Set the number of decomposition levels         |
| **Clear Button**        | Reset all inputs to defaults                   |

## Example

1. Upload an image.
2. Increase **Sigma** to add stronger noise.
3. Adjust **Wavelet** and **Level** to get better denoising.
4. Download noisy and denoised images.

## Notes

- Input images are internally handled as NumPy arrays (`float32` type).
- Output images are clipped between 0–255 and cast to `uint8`.
- Thresholding uses a universal threshold formula based on estimated noise standard deviation.

## Authors
### Haris Narrendran R   23B1857
### Shreyas Venkata Ramanan 23B1845
### Samridh Tiwari 23B1834
