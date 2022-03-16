import cherrypy
import json
import os
import adafruit_dht
import numpy as np
import time
import tensorflow as tf
from board import D4
import datetime
from mqtt_setup import setup
from datetime import datetime
import base64
import numpy as np 

class ADD(object):
    exposed = True

    def GET(self, *path):        
       pass

    ##################### Post is the suitable HTTP method  ####################
    def POST(self, *path, **query):

        ##### Adding models 
        body = cherrypy.request.body.read()
        body = json.loads(body)
        model_string = body.get('model')
        model_name = body.get('name')
        

        ##################### Control errors ####################
        if model_string is None: 
            raise cherrypy.HTTPError(400, 'Model empty')
        if model_name is None: 
            raise cherrypy.HTTPError(400, 'Model name empty')

        #################### ####################  ####################

        # receive the string, pass it to 64 decode
        model_base64 = base64.b64decode(model_string)

        if not os.path.exists('./models'):      ### create a directory model if not existed
                os.mkdir('models')

        path_file=f'./models/{model_name}.tflite'

        ##### SAVE THE MODEL AS .tflite
        with open(path_file , 'wb') as f:
            f.write(model_base64)

    def PUT(self, *path, **query):
            pass

    def DELETE(self, *path, **query):
        pass


############################################ CLASS LIST ############################################

class LIST(object):
    exposed = True

    def GET(self, *path):          
        pass

    def POST(self, *path, **query):

        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass
############################################ CLASS PREDICT ############################################
class PREDICT(object):
    exposed = True
    
    def GET(self, *path):         
        pass

    def POST(self, *path, **query):

        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass
              

                

if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(ADD(), '/add', conf)
    cherrypy.tree.mount(LIST(), '/list', conf)
    cherrypy.tree.mount(PREDICT(), '/predict', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()

    cherrypy.engine.block()


    
