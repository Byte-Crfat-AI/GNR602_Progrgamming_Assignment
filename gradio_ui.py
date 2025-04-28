import cv2
import numpy as np
import pywt
import gradio as gr
import matplotlib.pyplot as plt

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 255)

def wavelet_denoise(img, wavelet='db6', level=2):
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    cA, cD = coeffs[0], coeffs[1:]
    new_cD = []
    for detail in cD:
        subband_thresh = []
        for d in detail:
            sigma = np.median(np.abs(d)) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(d.size))
            subband_thresh.append(pywt.threshold(d, threshold, mode='soft'))
        new_cD.append(tuple(subband_thresh))
    new_coeffs = [cA] + new_cD
    return pywt.waverec2(new_coeffs, wavelet)

def process(image, sigma, wavelet='db6', level=2):
    img = np.array(image).astype(np.float32)
    R,G,B = cv2.split(img)
    R_sigma = np.std(R)
    G_sigma = np.std(G)
    B_sigma = np.std(B)
    # print("Before Noise addition")
    # print(f"R channel noise std: {R_sigma}, G channel noise std: {G_sigma}, B channel noise std: {B_sigma}")
    noisy_img = add_gaussian_noise(img, sigma)
    R_noisy, G_noisy, B_noisy = cv2.split(noisy_img)
    R_noisy_std = np.std(R_noisy)
    G_noisy_std = np.std(G_noisy)
    B_noisy_std = np.std(B_noisy)
    # print("After Noise addition")
    # print(f"R channel noise std: {R_noisy_std}, G channel noise std: {G_noisy_std}, B channel noise std: {B_noisy_std}")
    denoised_img_R = wavelet_denoise(R_noisy, wavelet=wavelet, level=level)
    denoised_img_G = wavelet_denoise(G_noisy, wavelet=wavelet, level=level)
    denoised_img_B = wavelet_denoise(B_noisy, wavelet=wavelet, level=level)
    denoised_img_R_std = np.std(denoised_img_R)
    denoised_img_G_std = np.std(denoised_img_G)
    denoised_img_B_std = np.std(denoised_img_B)
    # print("After Denoising")
    # print(f"R channel noise std: {denoised_img_R_std}, G channel noise std: {denoised_img_G_std}, B channel noise std: {denoised_img_B_std}")
    # print("__________________________________________________________________")
    denoised_img = cv2.merge((denoised_img_R, denoised_img_G, denoised_img_B))
    noisy_img_uint8 = np.clip(noisy_img, 0, 255).astype(np.uint8)
    denoised_img_uint8 = np.clip(denoised_img, 0, 255).astype(np.uint8)
    return noisy_img_uint8, denoised_img_uint8

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy")
            sigma_input = gr.Slider(0, 100, value=20, step=1, label="Sigma")
            wavelet_input = gr.Radio(value='db6', choices=["db1", "db2", "db3", "db4", "db5", "db6"], label="Wavelet Type")
            level_input = gr.Slider(1, 5, value=2, step=1, label="Decomposition Level")
            clear_btn = gr.Button("Clear")
        with gr.Column():
            noisy_output = gr.Image(type="numpy", label="Noisy Image", show_download_button = True)
            denoised_output = gr.Image(type="numpy", label="Smoothed Image",   show_download_button = True)

    inputs = [image_input, sigma_input, wavelet_input, level_input]
    outputs = [noisy_output, denoised_output]

    image_input.change(fn=process, inputs=inputs, outputs=outputs)
    sigma_input.change(fn=process, inputs=inputs, outputs=outputs)
    wavelet_input.change(fn=process, inputs=inputs, outputs=outputs)
    level_input.change(fn=process, inputs=inputs, outputs=outputs)

    def reset():
        return None, 20, 'db6', 2

    clear_btn.click(fn=reset, outputs=[image_input, sigma_input, wavelet_input, level_input])

iface.launch(share=True)
