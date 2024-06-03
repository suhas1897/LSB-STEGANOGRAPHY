# app1.py
from flask import Flask, request, render_template
import cv2
import math
import numpy as np

app1 = Flask(__name__)

# Embed secret in the n least significant bit.
# Lower n make picture less loss but lesser storage capacity.
BITS = 2

LOW_BITS = (1 << BITS) - 1
BYTES_PER_BYTE = math.ceil(8 / BITS)
FLAG = '%'


def extract(path):
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
    data = np.reshape(img, -1)
    total = data.shape[0]
    res = ''
    idx = 0
    # Decode message length
    while idx < total // BYTES_PER_BYTE:
        ch = decode(data[idx*BYTES_PER_BYTE: (idx+1)*BYTES_PER_BYTE])
        idx += 1
        if ch == FLAG:
            break
        res += ch
    end = int(res) + idx
    assert end <= total // BYTES_PER_BYTE, "Input image isn't correct."

    secret = ''
    while idx < end:
        secret += decode(data[idx*BYTES_PER_BYTE: (idx+1)*BYTES_PER_BYTE])
        idx += 1
        # print(secret)
    return secret


def decode(block):
    val = 0
    for idx in range(len(block)):
        val |= (block[idx] & LOW_BITS) << (idx * BITS)
    return chr(val)


@app1.route('/')
def index():
    return render_template('index2.html')


@app1.route('/extract', methods=['POST'])
def extract_message():
    file = request.files['file']
    file.save(file.filename)
    message = extract(file.filename)
    return message


if __name__ == '__main__':
    app1.run(debug=True, port = 8080)
