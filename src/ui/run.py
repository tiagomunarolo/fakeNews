from flask import Flask
from src.ui.routes.check import check_bp

app = Flask(__name__)
app.register_blueprint(check_bp, url_prefix='/check')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
