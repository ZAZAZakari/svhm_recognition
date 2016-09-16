import os
import predictor
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from predictor import Predictor

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
p = Predictor('streetNumberClassifier')
app = Flask(__name__, static_url_path='', static_folder='')
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/results/<filename>')
def uploaded_file(filename):
	print filename
	return '<!doctype html><img src="/uploads/' + filename + '"/><br/><img src="/predictions/' + filename + '"/>'
	
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('uploads', filename))
            p.predict(filename)
            return redirect(url_for('uploaded_file', filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
app.run(host='0.0.0.0', port='5002', threaded=True)