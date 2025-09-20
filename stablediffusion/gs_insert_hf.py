from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import numpy as np
from scipy.stats import norm
import torch
import os
from datetime import datetime

def gs_watermark_init_noise(shape=None, message="", key_hex="", nonce_hex="", 
                           device="cpu", dtype=torch.float32, opt=None):
    """
    Generate watermarked initial noise for Stable Diffusion
    
    This function works with both the original Stable Diffusion implementation
    and the new Hugging Face diffusers pipeline.
    
    Args:
        shape: Tuple specifying the shape of the latent tensor (B, C, H, W)
        message: Message to embed in the watermark
        key_hex: Encryption key as hex string
        nonce_hex: Nonce as hex string 
        device: PyTorch device (cpu/cuda)
        dtype: PyTorch data type
        opt: Legacy opt object for backward compatibility
    
    Returns:
        torch.Tensor: Watermarked noise tensor
    """
    
    # Handle legacy opt object format
    if opt is not None:
        key_hex = getattr(opt, 'key_hex', key_hex)
        nonce_hex = getattr(opt, 'nonce_hex', nonce_hex)
        # For legacy compatibility, assume 4x64x64 shape if not provided
        if shape is None:
            shape = (1, 4, 64, 64)
    
    # Default shape for SD 2.1 768x768 images
    if shape is None:
        shape = (1, 4, 96, 96)  # 768/8 = 96
    
    # Process message
    if message:
        # Convert message to bytes
        message_bytes = message.encode()
        # Ensure the encoded message is 256 bits (32 bytes)
        if len(message_bytes) < 32:
            padded_message = message_bytes + b'\x00' * (32 - len(message_bytes))
        else:
            padded_message = message_bytes[:32]
        k = padded_message
    else:
        # If message is empty, generate random 256-bit watermark message k
        k = os.urandom(32)
    
    # Diffusion process, replicate 64 times
    s_d = k * 64
    
    # Use ChaCha20 encryption
    if key_hex and nonce_hex:
        # Convert hex strings to bytes
        key = bytes.fromhex(key_hex)
        nonce = bytes.fromhex(nonce_hex)
    elif key_hex and not nonce_hex:
        # Use center 16 bytes of key as nonce
        key = bytes.fromhex(key_hex)
        nonce_hex_part = key_hex[16:48]
        nonce = bytes.fromhex(nonce_hex_part)
    else:
        key = os.urandom(32)
        nonce = os.urandom(16)
    
    # Encrypt
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    m = encryptor.update(s_d) + encryptor.finalize()
    
    # Convert m to binary form, m follows uniform distribution
    m_bits = ''.join(format(byte, '08b') for byte in m)
    
    # Initialize result array
    batch_size, channels, height, width = shape
    total_elements = channels * height * width
    
    # Window size l, can be values other than 1
    l = 1
    
    # Create watermarked noise array
    watermark_array = np.zeros((channels, height, width))
    
    index = 0
    # Traverse m's binary representation, cutting according to window size l
    for i in range(0, len(m_bits), l):
        if index >= total_elements:
            break
            
        window = m_bits[i:i+l]
        y = int(window, 2)  # Convert binary sequence in window to integer y
        
        # Generate random u
        u = np.random.uniform(0, 1)
        
        # Calculate z^s_T
        z_s_T = norm.ppf((u + y) / 2**l)
        
        # Map to 3D coordinates
        c = index // (height * width)
        h = (index // width) % height
        w = index % width
        
        if c < channels:
            watermark_array[c, h, w] = z_s_T
        
        index += 1
    
    # Convert to PyTorch tensor
    watermark_tensor = torch.from_numpy(watermark_array).unsqueeze(0).to(device=device, dtype=dtype)
    
    # Log watermark information
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('info_data.txt', 'a') as f:
        f.write(f"Time: {current_time}\n")
        f.write(f'key: {key.hex()}\n')
        f.write(f'nonce: {nonce.hex()}\n')
        f.write(f'message: {k.hex()}\n')
        f.write(f'shape: {shape}\n')
        f.write('----------------------\n')
    
    return watermark_tensor

# Legacy function for backward compatibility
def gs_watermark_init_noise_legacy(opt, message=""):
    """Legacy wrapper for backward compatibility with original implementation"""
    return gs_watermark_init_noise(
        shape=(1, 4, 64, 64),  # Original default shape
        message=message,
        opt=opt
    )