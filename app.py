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
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

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
                "text": "Please convert Invoice data into text i want all the informations in arabic, i want the first name, last name, id number and date of birth in json format  inside '{' and '}'",
            },
            {"type": "image_url", "image_url": image_path},
        ]
    )

    message = llm.invoke([message])

    #return message.content
    print(message.content)

    start_index = message.content.find("{") + 1
    end_index = message.content.find("}")
    json_text = message.content[start_index:end_index]
    print("json text -----> ",json_text)
    print(type(json_text))
    str_dict = str(json_text)
    j_format = str_dict.replace("'", "\"")
    print("j_format -->",j_format)

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    translate_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"Please translate the following text into French, with the same the JSON format. Also, the Arabic 'nom' and 'prenom' to French :\n{j_format}",
            },
        ]
    )

    #create the humanmassage propmt templete with the image file
    '''translate_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Please translate the following json to French (literal translation, not meaning-based):",
            },
            {"type": "json", "data": p_data},
        ]
    )'''

    #chat_model = ChatGoogleGenerativeAI(model_name="gemini-pro", google_api_key=api_key)

    # Invoke the LLM for translation
    response = llm.invoke([translate_message])
    print(response.content)

    # Extract the translated text from the response
    #translated_text = response["completions"][0]["text"]


    # Translate from French to Arabic
    #translated_text = GoogleTranslator(source='arabic', target='french').translate(str(json_text))
    #print(translated_text)

    # Parse the LaMDA response (assuming JSON format)
    try:
        parsed_data = json.loads(response.content)
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
@app.route('/congrats', methods=['POST'])
def congratulations():
    first_name = request.form.get('hidden_prenom')
    last_name = request.form.get('hidden_nom')
    id_number = request.form.get('hidden_numero')
    date_of_birth = request.form.get('date_de_naissance')
    address = request.form.get('adresse')
    postal_code = request.form.get('postal_code')
    city = request.form.get('ville')
    return render_template('congrats.html', first_name=first_name,last_name=last_name,id_number=id_number,date_of_birth=date_of_birth,address=address,postal_code=postal_code,city=city)


@app.route('/invoice', methods=['POST'])
def invoice():
    first_name = request.form.get('hidden_prenom')
    last_name = request.form.get('hidden_nom')
    id_number = request.form.get('hidden_numero')
    address = request.form.get('hidden_adresse')
    postal_code = request.form.get('hidden_postal_code')
    city = request.form.get('hidden_ville')
    return render_template('invoice.html', first_name=first_name,last_name=last_name,id_number=id_number,address=address,postal_code=postal_code,city=city)
if __name__ == '__main__':
    app.run(debug=True)


