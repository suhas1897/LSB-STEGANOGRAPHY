import math
from os import path
from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

BITS = 2
HIGH_BITS = 256 - (1 << BITS)
LOW_BITS = (1 << BITS) - 1
BYTES_PER_BYTE = math.ceil(8 / BITS)
FLAG = '%'

# Function to insert message into image
def insert(img_path, msg):
    img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
    ori_shape = img.shape
    max_bytes = ori_shape[0] * ori_shape[1] // BYTES_PER_BYTE
    msg = '{}{}{}'.format(len(msg), FLAG, msg)
    assert max_bytes >= len(msg), "Message greater than capacity:{}".format(max_bytes)
    data = np.reshape(img, -1)
    for (idx, val) in enumerate(msg):
        encode(data[idx*BYTES_PER_BYTE: (idx+1) * BYTES_PER_BYTE], val)

    img = np.reshape(data, ori_shape)
    filename, _ = path.splitext(img_path)
    filename = 'H:\\' + filename + '_lsb_embedded' + ".png"  # Save to H drive
    cv2.imwrite(filename, img)
    return filename

# Function to encode the message into image
def encode(block, data):
    data = ord(data)
    for idx in range(len(block)):
        block[idx] &= HIGH_BITS
        block[idx] |= (data >> (BITS * idx)) & LOW_BITS

# Home route
@app.route('/')
def index():
    return render_template('app.html')
    #return render_template('style.css')


# Endpoint to handle embedding message
@app.route('/embed', methods=['POST'])
def embed():
    image_file = request.files['image']
    message = request.form['message']
    filename = 'uploaded_image.png'
    image_file.save(filename)
    result_filename = insert(filename, message)
    return send_file(result_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
