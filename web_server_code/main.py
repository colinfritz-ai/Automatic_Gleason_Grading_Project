
import os
import sys
from PIL import Image
import tensorflow as tf
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
import tensorflow_io as tfio
import numpy as np 

app = Flask(__name__)

def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    pass


# Initialize our ngrok settings into Flask
app.config.from_mapping(
    BASE_URL="http://localhost:5000",
    USE_NGROK=os.environ.get("USE_NGROK", "False") == "True" and os.environ.get("WERKZEUG_RUN_MAIN") != "True"
)

if app.config.get("ENV") == "development" and app.config["USE_NGROK"]:
    print("public url updated")
    # pyngrok will only be installed, and should only ever be initialized, in a dev environment
    from pyngrok import ngrok

    # Get the dev server port (defaults to 5000 for Flask, can be overridden with `--port`
    # when starting the server
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 5000

    # Open a ngrok tunnel to the dev server
    public_url = ngrok.connect(port).public_url
    print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

    # Update any base URLs or webhooks to use the public ngrok URL
    app.config["BASE_URL"] = public_url
    init_webhooks(public_url)

app.config["UPLOAD_FOLDER"] = "uploads"


@app.route("/")
def index():
    return redirect("/static/index.html")


@app.route("/sendfile", methods=["POST"])
def send_file():
    fileob = request.files["file2upload"]
    filename = secure_filename(fileob.filename)
    save_path = "{}/{}".format(app.config["UPLOAD_FOLDER"], filename)
    fileob.save(save_path)

    # open and close to update the access time.
    with open(save_path, "r") as f:
        pass

    return "successful_upload"


@app.route("/filenames", methods=["GET"])
def get_filenames():
    filenames = os.listdir("uploads/")

    def modify_time_sort(file_name):
        file_path = "uploads/{}".format(file_name)
        file_stats = os.stat(file_path)
        last_access_time = file_stats.st_atime
        return last_access_time

    filenames = sorted(filenames, key=modify_time_sort)
    print("sorted_names" + str(filenames))
    return_dict = dict(filenames=filenames)
    print("sorted dictionary:" + str(return_dict))
    def predict_on_file(times_uploaded):
        #must produce a list with key filenames to work.  
        #otherwise should function with the prediction code.
        predictions = []  
        save_path = "/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/train_production"
        model = tf.keras.models.load_model(save_path)
        for key, value in  times_uploaded.items():
            file_path = value[1:]
            for image in file_path:
                img=tf.io.read_file("uploads/"+image)
                img=tfio.experimental.image.decode_tiff(img)
                img=img[:,:,0:3]
                img=tf.image.resize(img, size = (512,512))
                img = tf.expand_dims(img, axis =0)
                prediction=model.predict(img)
                prediction=np.argmax(prediction)
                predictions.append(str(prediction))
        times_uploaded[key] = predictions
        return times_uploaded

    return_dict=predict_on_file(return_dict)
    print("final return dict getting close: " + str(return_dict))
    return jsonify(return_dict)


if __name__ == '__main__':
    app.run(debug=False)
