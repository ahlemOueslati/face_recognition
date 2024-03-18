import sub as sub
from flask import Flask, request, redirect, url_for, send_from_directory, render_template,Response
import face_recognition
from PIL import Image
import numpy as np
import os
import cv2
import time

app = Flask(__name__)

extracted_face = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global extracted_face
    if request.method == 'POST':
        if 'id-card-upload' not in request.files:
            return redirect(request.url)

        file = request.files['id-card-upload']

        if file.filename != '':
            filename = file.filename
            file.save(os.path.join('images', filename))
            extracted_face = get_face(os.path.join('images', filename))
            return render_template('index.html', extracted_face=extracted_face)

    return render_template('index.html', extracted_face=extracted_face)

def get_face(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')  # Convert to RGB format
    face_image_np = np.array(image)
    face_locations = face_recognition.face_locations(face_image_np)

    if len(face_locations) > 0:
        top, right, bottom, left = face_locations[0]
        face_image = Image.fromarray(face_image_np[top:bottom, left:right])
        extracted_face_filename = os.path.join('images', 'extracted_face.jpg')
        face_image.save(extracted_face_filename)
        return extracted_face_filename

    return None

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    extracted_face_path = os.path.join('images', 'extracted_face.jpg')
    if not os.path.exists(extracted_face_path):
        return "Error: Extracted face not found. Please upload your ID card first."

    extracted_face_image = face_recognition.load_image_file(extracted_face_path)
    extracted_face_encoding = face_recognition.face_encodings(extracted_face_image)[0]

    video_capture = cv2.VideoCapture(0)

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (_, right, _, left), face_encoding in zip(face_locations, face_encodings):
                match_results = face_recognition.compare_faces([extracted_face_encoding], face_encoding)
                if True in match_results:
                    match_score = face_recognition.face_distance([extracted_face_encoding], face_encoding)[0]
                    return render_template('match.html', match_score=match_score)
                else:
                    return redirect(url_for('no_match'))

    return redirect(url_for('index'))


@app.route('/match')
def match():
    return render_template('match.html')

@app.route('/no_match')
def no_match():
    return render_template('no_match.html')

@app.route('/face',methods=['GET', 'POST'])
def get_camera():
    """Video streaming home page."""
    return render_template('face.html')
def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else:
            break
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
