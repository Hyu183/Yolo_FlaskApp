from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
import os
import yolo_detector
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


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

        yolo_detector.yolo_detector(filepath,filename)
    
    return render_template('uploaded.html', display_detection = filename, fname = filename)


if __name__ == "__main__":
    app.run(debug=True)
