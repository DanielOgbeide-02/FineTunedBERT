from flask import Flask
app = Flask(__name__)

for rule in app.url_map.iter_rules():
    print(rule)
