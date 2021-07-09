from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
import os
import yolo_detector
import bird_detector

for f in os.listdir("static\\detections\\"):
    os.remove("static\\detections\\"+f)

app = Flask(__name__)


model = None
model_type = 0


@app.route('/')
def index():
    global model 
    global model_type
    if(model_type == 1 or model == None ):
        model_type = 0
        with open("./data/model_type.txt", 'w') as f:
            f.write(str(model_type))
        model = yolo_detector.load_yolo_model()
        
    return render_template('index.html')

@app.route('/bird_detection')
def bird_detection():
    global model 
    global model_type
    if(model_type == 0 or model == None ):
        model_type = 1
        with open("./data/model_type.txt", 'w') as f:
            f.write(str(model_type))  
        model = bird_detector.load_bird_model()
        
    return render_template('bird_detection.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # create a secure filename
        filename = secure_filename(f.filename)
        print(filename)
        # save file to /static/uploads
        basepath = os.path.dirname(__file__) + '\static'
        print(basepath)
        filepath = os.path.join( basepath, 'uploads', filename)
        print(filepath)
        f.save(filepath)

        global model 
        global model_type

        if(model_type == 0):
            yolo_detector.yolo_detector(filepath,filename,model)
        elif(model_type== 1):
            bird_detector.bird_detector(filepath,filename,model)
        
        print('Detection obj complete!!!')
    return render_template('uploaded.html', display_detection = filename, fname = filename)


if __name__ == "__main__":
    app.run(debug=True)
