import json
import os
import re
import time

import cv2
import face_recognition
import numpy as np
from PIL import Image
from flask import Flask, request, session, redirect, url_for, send_from_directory, render_template, Response
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)
app.secret_key = 'hello'
filename = None
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
            session['filename']=filename
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

    #return redirect(url_for('extract', parsed_data=extracted_text))
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


# ... Hana code

def parse_json(input_string):
    # Split the input string into lines
    lines = input_string.strip().split("\n")

    # Define regular expressions to match the required fields
    regex_numero = r"\b(\d{8})\b"
    regex_prenom = r"الاسم\s+(.+)"
    regex_nom = r"اللقب\s+(.+)"

    # Initialize the dictionary
    result_dict = {}

    # Iterate through the lines and extract the required fields
    for line in lines:
        match_numero = re.search(regex_numero, line)
        match_prenom = re.search(regex_prenom, line)
        match_nom = re.search(regex_nom, line)

        if match_numero:
            result_dict["numero"] = match_numero.group(1)
        elif match_prenom:
            result_dict["prenom"] = match_prenom.group(1)
        elif match_nom:
            result_dict["nom"] = match_nom.group(1)

    # Extract date de naissance
    for line in lines:
        date_naissance = re.search(r"تاريخ الولادة (\d{1,2}) (.+) (\d{4})", line)
        if date_naissance:
            result_dict["date_de_naissance"] = f"{date_naissance.group(2)} {date_naissance.group(3)} {date_naissance.group(1)}"

    return result_dict

def get_information(image_path):
    """Extracts information from the ID card image using LaMDA."""
    api_key = "AIzaSyBvgpQrYcLIaQmFPsaU0D0NDjWX4endWqg"  # Access from environment variable

    if not api_key:
        print("Error: Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
        return None

 

    llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", google_api_key=api_key)

    #create the humanmassage propmt templete with the image file
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Convert Invoice data into text i want all the informations in arabic   ",
            },
            {"type": "image_url", "image_url": image_path},
        ]
    )

    message = llm.invoke([message])

    #return message.content
    print(message.content)

    json_text  = parse_json(message.content)
    print(json_text)

    #create the humanmassage propmt templete with the image file
    translate_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Please translate the following text to French (literal translation, not meaning-based):",
            },
            {"type": "text", "text": json_text},
        ]
    )

    # Invoke the LLM for translation
    response = llm.invoke([translate_message])

    translated_text = response.results[0].completion


    # Translate from French to Arabic
    #translated_text = GoogleTranslator(source='arabic', target='french').translate(str(json_text))
    print(translated_text)

    # Parse the LaMDA response (assuming JSON format)
    try:
        print(json_text)
        parsed_data = json.loads(str(json_text))
        return parsed_data
    except json.JSONDecodeError:
        print("Error: LaMDA response could not be parsed as JSON.")
        return None, None, None

    return redirect(request.url,message.content)
@app.route('/extract')
def extract():
    filename = session.get('filename', 'Default value if not found')

    extracted_text = get_information(os.path.join('images', filename))
    return render_template('extract_information.html', parsed_data=extracted_text)

if __name__ == '__main__':
    app.run(debug=True)


