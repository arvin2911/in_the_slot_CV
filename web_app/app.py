from flask import Flask,request, url_for, redirect, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename 
import os
import subprocess
import EE399_MNIST
# import pickle
# import numpy as np

# start Flask
app = Flask(__name__)

# add secret key (to see the form in our template)
app.config['SECRET_KEY'] = 'umpire_training_app'
# upload destination folder
app.config['UPLOAD_FOLDER'] = 'static/files'

# upload file
class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")

# model=pickle.load(open('model.pkl','rb'))

# open route to html template
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)   # Obtain file path
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) # Obtain file path 
            file.save(file_path) # Save file into destination folder
            
            # Run the Python file with the uploaded video file as input
            # output = execute_python_script(file_path)
            output = EE399_MNIST.process_video(file_path)
            
            return render_template('result.html', output=output)
        else:
            return "Invalid file format. Only video files allowed."
    return render_template('index.html', form=form)

# @app.route('/predict',methods=['POST','GET'])
# def predict():
#     int_features=[int(x) for x in request.form.values()]
#     final=[np.array(int_features)]
#     print(int_features)
#     print(final)
#     prediction=model.predict_proba(final)
#     output='{0:.{1}f}'.format(prediction[0][1], 2)

#     if output>str(0.5):
#         return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
#     else:
#         return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")

# checks if the file has the allowed extension or type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'jpg', 'png'}

# executes a python program using the file's path
def execute_python_script(video_path):
    try:
        # Execute the Python script with the video file path as argument
        output = subprocess.check_output(['python', 'models/demo.py', video_path], stderr=subprocess.STDOUT, universal_newlines=True)
        # Assuming that the output of EE399_MNIST.py is the path to the generated picture or video
        result_path = output.strip()  # Remove any leading/trailing whitespace
        return result_path
        
    except subprocess.CalledProcessError as e:
        return "Error: {}".format(e.output)


# run the app
if __name__ == "__main__":
    app.run(debug=True)