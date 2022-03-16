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

  
    def POST(self, *path, **query):

        pass

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


    
