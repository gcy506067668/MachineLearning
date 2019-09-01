from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/")
def index():
    return "<form name=\"input\" action=\"/login\" method=\"post\"> Username: <input type=\"text\" name=\"username\">Password: <input type=\"text\" name=\"password\"> <input type=\"submit\" value=\"Submit\"> </form>"



@app.route("/login",methods=("GET", "POST"))
def login():
    username = request.form["username"]
    password = request.form["password"]
    print(username)
    print(password)
    return "login user:"+username

if __name__ == '__main__':
    app.run()