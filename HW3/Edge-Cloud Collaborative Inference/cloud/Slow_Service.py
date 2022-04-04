import cherrypy
import json
import json
import base64
import numpy as np
import tensorflow as tf
import re
import os
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
############################### Reading the testsplit and labels.txt ###############################
zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')

test_files = np.loadtxt("kws_test_split.txt" , dtype = str )
labels = np.loadtxt("labels.txt" , dtype = "object" ,delimiter= "," )

labels = [re.sub("[]''[]","", x) for x in labels]
labels = [re.sub("'","", x.strip()) for x in labels]
labels = np.array(labels , dtype = str) 
labels =  tf.convert_to_tensor(labels)

############ Create the Keywords Spotting Class KWS ######################3
class  KWS(object):
    exposed = True

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):

        pass
    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(KWS(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
