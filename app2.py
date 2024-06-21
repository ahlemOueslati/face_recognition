import json
import os
import re
import time
import cv2
import face_recognition
import numpy as np
from PIL import Image
from flask import Flask, request, session, redirect, url_for, send_from_directory, render_template, Response
#from langchain_core.messages import HumanMessage
#from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from huggingface_hub import HfApi, HfFolder
import numpy as np
from PIL import Image
import requests
import torch 
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration,BitsAndBytesConfig
import argparse

app = Flask(__name__)
app.secret_key = 'hello'
filename = None
extracted_face = None
device=None
model=None

processor=None
token = "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"
image=None

def load_model_and_tokenizer(strategy, model_path="google/paligemma-3b-mix-224"):
    global model
    global processor
    global token

    print("Loading the Model")
    if not processor:
        processor= AutoProcessor.from_pretrained(model_path)
    
    if not model:
        if strategy == "cpu":
            model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, token=token)
        elif strategy == "gpu":
            print("GPU available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("Current device:", torch.cuda.current_device())
                print("Device name:", torch.cuda.get_device_name())
                print("Device capability:", torch.cuda.get_device_capability())
                print("Device memory (GB):", torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3))

            model = PaliGemmaForConditionalGeneration.from_pretrained(model_path,token=token)
        elif strategy == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, quantization_config=quantization_config, token=token)
        elif strategy == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, quantization_config=quantization_config, token=token)
        elif strategy == "flash":
            model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2", token=token).to("cuda:0")
        else:
            raise ValueError("Invalid strategy specified")
    model.eval()  # Set the model to evaluation mode
    print("Done loading the model...")
    return model, processor

def generate_text(prompt,image, model,processor, max_length=512, temperature=0.8):
    model_inputs = processor(prompt,images=image, return_tensors='pt')
    input_len=model_inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_length=max_length, temperature=temperature)
        generation=generation[0][input_len:]
        text=processor.decode(generation, skip_special_tokens=True)
    return  text



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
    global device
    global lines
    global out
    #from huggingface_hub import HfApi, HfFolder

    # Provide your token here
    # Save the token
    #HfFolder.save_token(token)

    # Verify by logging in using HfApi
    #api = HfApi()
    #api.whoami()

    #model_id = "google/paligemma-3b-mix-224"

    # Load an image from a URL
    url = image_path
    image = Image.open(url).convert('RGB')
    

    # Load the model and processor
    #model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
    #processor = AutoProcessor.from_pretrained(model_id)

    # Instruct the model to create a caption in Arabic 
    prompt = "OCR"
    model, processor = load_model_and_tokenizer(args.strategy)
    text = generate_text(prompt,image, model, processor)
    lines = text.split('\n')   
    # Extract fields from lines list
    print(text)
    print(lines)
    first_name = lines[-4] if len(lines) > 0 else ''
    print("first_name",first_name)

    last_name = lines[3] if len(lines) > 0 else ''
    id_number = lines[2] if len(lines) > 0 else ''
    print("id_number",id_number)
    address =  ''
    import re

    # Regular expression pattern to match the date format
    date_pattern = r' \d{2} [ء-ي]+ \d{4}'


    # Find the date in the OCR text
    match = re.search(date_pattern, text)
   
    extracted_date = match.group(0)           
    print(f"Extracted Date: {extracted_date}")
    
    date_of_birth= extracted_date if len(lines) > 0 else ''
    #Postal code here is the name of his fatehr and grandfatheri'll implement later 
    postal_code = lines[-3] if len(lines) > 0 else ''
    city = ''
    family_name = lines[-3] if len(lines) > 0 else ''
    # Translate from French to Arabic (example)
    #translated_text = GoogleTranslator(source='french', target='arabic').translate(text)
    print("first_name",first_name)
    print("last_name",last_name)
    print("id_number",id_number)
    print("date_of_birth",date_of_birth)
    print("postal_code",postal_code)
    print("family_name",family_name)

    out= {
        'first_name': first_name,
        'last_name': last_name,
        'id_number': id_number,
        'date_of_birth': date_of_birth,
        'address': address,
        'postal_code': postal_code,
        'city': city,
        'family_name':family_name
        
    }
    

    

    # Extract the translated text from the response
    #translated_text = response["completions"][0]["text"]


    # Translate from French to Arabic
    #translated_text = GoogleTranslator(source='arabic', target='french').translate(str(json_text))
    #print(translated_text)
     #,redirect(request.url,out)
   
     
    return out
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
    family_name=request.form.get('family_name')
    return render_template('congrats.html', first_name=first_name,last_name=last_name,id_number=id_number,date_of_birth=date_of_birth,address=address,postal_code=postal_code,city=city,family_name=family_name)


@app.route('/invoice', methods=['POST'])
def invoice():
    first_name = request.form.get('hidden_prenom')
    last_name = request.form.get('hidden_nom')
    id_number = request.form.get('hidden_numero')
    address = request.form.get('hidden_adresse')
    postal_code = request.form.get('hidden_postal_code')
    city = request.form.get('hidden_ville')
    family_name=request.form.get('hidden_family_name')
    return render_template('invoice.html', first_name=first_name,last_name=last_name,id_number=id_number,address=address,postal_code=postal_code,city=city,family_name=family_name)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Flask server with different model loading strategies")
    parser.add_argument('--strategy', type=str, choices=['cpu', 'gpu', '8bit', '4bit', 'flash'], required=True, help='The strategy to load the model')
    args = parser.parse_args()
    print("Listening...")
    app.run(debug=True)
