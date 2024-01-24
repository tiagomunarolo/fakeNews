from flask import Blueprint

check_bp = Blueprint(name='check_news',
                     import_name=__name__,
                     template_folder='templates')


@check_bp.route('/', methods=['GET'])
def check_news():
    return "OK"
