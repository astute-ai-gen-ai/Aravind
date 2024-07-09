from flask import Flask

def create_app():
    app=Flask(__name__)
    app.config['SECRET_KEY'] = 'backend_astute_ai'

    from app.routes import routes
    app.register_blueprint(routes) 
    
    return app