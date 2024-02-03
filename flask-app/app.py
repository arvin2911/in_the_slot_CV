from flask import Flask, render_template
import urllib.request

def is_it_raining_in_seattle():
    with urllib.request.urlopen('https://depts.washington.edu/ledlab/teaching/is-it-raining-in-seattle/') as response:
        is_it_raining_in_seattle = response.read().decode()

    if is_it_raining_in_seattle == "true":
        return True
    else:
        return False


app = Flask(__name__)

# Without template
# @app.route("/")
# def index():
#     if is_it_raining_in_seattle():
#         return "<h1>Yes</h1>"
#     else:
#         return "<h1>No</h1>"


# With template
@app.route("/")
def index():
    if is_it_raining_in_seattle():
        is_it_raining = "Yes"
    else:
        is_it_raining = "No"
    
    # Note how the parameter name is the same as what is in the
    # template file (index.html)
    return render_template("index.html", raining=is_it_raining)
