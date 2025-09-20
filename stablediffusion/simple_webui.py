#!/usr/bin/env python3
"""
Simple Flask Web Interface for Watermarked Stable Diffusion
A lightweight alternative to the full SD-WebUI
"""

import os
import sys
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify
import json

app = Flask(__name__)

# Configuration
PYTHON_EXE = r"C:/Users/prachi pandey/Desktop/Guassian/.venv/Scripts/python.exe"
WORK_DIR = r"C:\Users\prachi pandey\Desktop\Guassian\stablediffusion"

@app.route('/')
def index():
    """Main page with generation form"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Watermarked Stable Diffusion WebUI</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
        textarea { height: 80px; resize: vertical; }
        .row { display: flex; gap: 20px; }
        .col { flex: 1; }
        button { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        button:disabled { background: #cccccc; cursor: not-allowed; }
        .status { margin-top: 20px; padding: 15px; border-radius: 5px; min-height: 100px; }
        .status.loading { background: #e3f2fd; border: 1px solid #2196F3; }
        .status.success { background: #e8f5e8; border: 1px solid #4CAF50; }
        .status.error { background: #ffebee; border: 1px solid #f44336; }
        .result-image { max-width: 100%; margin-top: 20px; border-radius: 5px; }
        .watermark-section { background: #f9f9f9; padding: 20px; border-radius: 5px; margin-top: 20px; }
        .advanced { margin-top: 10px; }
        .toggle { cursor: pointer; color: #2196F3; text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Watermarked Stable Diffusion WebUI</h1>
        </div>
        
        <form id="generateForm">
            <div class="form-group">
                <label for="prompt">Prompt:</label>
                <textarea id="prompt" name="prompt" placeholder="Enter your image description here..." required>a beautiful landscape with mountains and lake</textarea>
            </div>
            
            <div class="row">
                <div class="col">
                    <div class="form-group">
                        <label for="width">Width:</label>
                        <select id="width" name="width">
                            <option value="512" selected>512</option>
                            <option value="768">768</option>
                            <option value="1024">1024</option>
                        </select>
                    </div>
                </div>
                <div class="col">
                    <div class="form-group">
                        <label for="height">Height:</label>
                        <select id="height" name="height">
                            <option value="512" selected>512</option>
                            <option value="768">768</option>
                            <option value="1024">1024</option>
                        </select>
                    </div>
                </div>
                <div class="col">
                    <div class="form-group">
                        <label for="steps">Steps:</label>
                        <select id="steps" name="steps">
                            <option value="15">15 (Fast)</option>
                            <option value="20" selected>20 (Good)</option>
                            <option value="30">30 (High Quality)</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="watermark-section">
                <h3>Watermark Settings</h3>
                <div class="form-group">
                    <label for="message">Watermark Message:</label>
                    <input type="text" id="message" name="message" placeholder="Enter watermark text..." value="MyWatermark2025">
                </div>
                
                <div class="toggle" onclick="toggleAdvanced()">Advanced Settings (click to expand)</div>
                <div id="advanced" class="advanced" style="display: none;">
                    <div class="form-group">
                        <label for="key_hex">Key (Hex) - Leave empty for default:</label>
                        <input type="text" id="key_hex" name="key_hex" placeholder="32-byte hex key...">
                    </div>
                    <div class="form-group">
                        <label for="nonce_hex">Nonce (Hex) - Leave empty for default:</label>
                        <input type="text" id="nonce_hex" name="nonce_hex" placeholder="16-byte hex nonce...">
                    </div>
                    <div class="form-group">
                        <label for="seed">Seed (0 for random):</label>
                        <input type="number" id="seed" name="seed" value="0">
                    </div>
                </div>
            </div>
            
            <div class="form-group">
                <button type="submit" id="generateBtn">Generate Watermarked Image</button>
            </div>
        </form>
        
        <div id="status" class="status" style="display: none;"></div>
        <div id="result" style="display: none;">
            <img id="resultImage" class="result-image" alt="Generated image">
        </div>
    </div>
    
    <script>
        function toggleAdvanced() {
            const advanced = document.getElementById('advanced');
            advanced.style.display = advanced.style.display === 'none' ? 'block' : 'none';
        }
        
        document.getElementById('generateForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            // Show loading status
            const statusDiv = document.getElementById('status');
            const resultDiv = document.getElementById('result');
            const generateBtn = document.getElementById('generateBtn');
            
            statusDiv.className = 'status loading';
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = 'Generating watermarked image... This may take 5-10 seconds.';
            resultDiv.style.display = 'none';
            generateBtn.disabled = true;
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    statusDiv.className = 'status success';
                    statusDiv.innerHTML = `Generation completed successfully!<br>Watermark: "${result.message}"<br>File: ${result.filename}<br>Time: ${result.generation_time}`;
                    
                    document.getElementById('resultImage').src = '/image/' + result.filename;
                    resultDiv.style.display = 'block';
                } else {
                    statusDiv.className = 'status error';
                    statusDiv.innerHTML = `Generation failed:<br>${result.error}`;
                }
            } catch (error) {
                statusDiv.className = 'status error';
                statusDiv.innerHTML = `Network error: ${error.message}`;
            }
            
            generateBtn.disabled = false;
        });
    </script>
</body>
</html>
    '''

@app.route('/generate', methods=['POST'])
def generate():
    """Handle image generation request"""
    try:
        data = request.json
        
        # Extract parameters
        prompt = data.get('prompt', '')
        message = data.get('message', 'default_message')
        width = int(data.get('width', 512))
        height = int(data.get('height', 512))
        steps = int(data.get('steps', 20))
        seed = int(data.get('seed', 0))
        key_hex = data.get('key_hex', '').strip()
        nonce_hex = data.get('nonce_hex', '').strip()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"webui_gen_{timestamp}.png"
        
        # Use defaults if not provided
        if not key_hex:
            key_hex = "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"
        if not nonce_hex:
            nonce_hex = "05072fd1c2265f6f2e2a4080a2bfbdd8"
        
        # Build command
        cmd = [
            PYTHON_EXE,
            "txt2img_hf_fixed.py",
            "--prompt", prompt,
            "--output", filename,
            "--message", message,
            "--key_hex", key_hex,
            "--nonce_hex", nonce_hex,
            "--width", str(width),
            "--height", str(height),
            "--steps", str(steps)
        ]
        
        if seed > 0:
            cmd.extend(["--seed", str(seed)])
        
        # Run generation
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORK_DIR)
        end_time = datetime.now()
        generation_time = f"{(end_time - start_time).total_seconds():.1f}s"
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'filename': filename,
                'message': message,
                'generation_time': generation_time
            })
        else:
            return jsonify({
                'success': False,
                'error': result.stderr or result.stdout
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/image/<filename>')
def serve_image(filename):
    """Serve generated images"""
    image_path = os.path.join(WORK_DIR, filename)
    if os.path.exists(image_path):
        return send_file(image_path)
    else:
        return "Image not found", 404

if __name__ == '__main__':
    print("Starting Watermarked Stable Diffusion WebUI")
    print("=" * 50)
    print(f"Working directory: {WORK_DIR}")
    print("WebUI will be available at: http://localhost:5000")
    print("GPU: NVIDIA RTX 3050 6GB (Ready!)")
    print("=" * 50)
    
    # Change to working directory
    os.chdir(WORK_DIR)
    
    # Start the web server
    app.run(host='127.0.0.1', port=5000, debug=False)