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

    ##################### GET is the suitable HTTP method  ####################
    def GET(self, *path):
        out = {}
        
        if not os.path.exists('./models'):
            print("send error code")
            raise cherrypy.HTTPError(400, 'Directory ./models missing')
        else:
            models=[]
            for x in os.listdir('./models'):
                if x.endswith(".tflite"):
                    models.append(x)
            out={'models':models}

        output = json.dumps(out)
        return output


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

############################################ CLASS PREDICT ############################################
class PREDICT(object):
    exposed = True
    def GET(self, *path, **query):

        # Read the query and extract the values + Managing Errors 
        #  model_name 
        model_name = query.get('model')
        if model_name is None:
            raise cherrypy.HTTPError(400, 'model_name missing')
        # tthres    
        tthres = query.get('tthres')
        if tthres is None:
            raise cherrypy.HTTPError(400, 'tthres missing')
        else:
            tthres = float(tthres)
        # hthres
        hthres = query.get('hthres')
        
        if hthres is None:
            raise cherrypy.HTTPError(400, 'hthres missing')
        else:
            hthres = float(hthres)

        test = setup("publisher 3")
        test.run()
        

        print(f'excuting model {model_name}')
        
        # depending on the model selected, read the model by the interpreter and allocate memory for tensors
        interpreter = tf.lite.Interpreter(model_path='./models/{}.tflite'.format(model_name))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        window = np.zeros([1, 6, 2], dtype=np.float32)
        expected = np.zeros(2, dtype=np.float32)

        MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
        STD = np.array([ 8.654227, 16.557089], dtype=np.float32)

        
        dht_device = adafruit_dht.DHT11(D4)
        first = True

        while True:

            if first == True : 
                for i in range(7):
                    try:
                        temperature = dht_device.temperature    # save the temperature readings to variable 
                        humidity = dht_device.humidity          # save the humidity readings to variable 
                    except RuntimeError as error:
                        # Errors happen fairly often, DHT's are hard to read, just keep going
                        print(f"Sensor Error {error.args[0]}")
                        time.sleep(3.0)
                        continue
                    except Exception as error:
                        dht_device.exit()
                        raise error
                    # the first 6 readings represent the window 
                    if i < 6:
                        window[0, i, 0] = np.float32(temperature)
                        window[0, i, 1] = np.float32(humidity)
                    # the seventh reading is the label 
                    if i == 6:
                        expected[0] = np.float32(temperature)
                        expected[1] = np.float32(humidity)

            
                        window = (window - MEAN) / STD            # Normalize the values for the window 
                
                        interpreter.set_tensor(input_details[0]['index'], window)
                        interpreter.invoke()
                        predicted = interpreter.get_tensor(output_details[0]['index'])
                        last_expected = np.array([expected[0] , expected[1]]  , dtype=np.float32 ,ndmin=3)
                        last_expected = (last_expected - MEAN) / STD
                        previous_window = window[:,1:,:]
                        first = False
            else :
                
                 
                temperature = dht_device.temperature
                humidity = dht_device.humidity
                expected[0] = np.float32(temperature)
                expected[1] = np.float32(humidity)
                window = np.append(previous_window , last_expected , axis=1)
                    
                #window = (window - MEAN) / STD
            
                interpreter.set_tensor(input_details[0]['index'], window)
                interpreter.invoke()
                predicted = interpreter.get_tensor(output_details[0]['index'])
            
                previous_window = window[:,1:,:]
                last_expected = np.array([expected[0] , expected[1]]  , dtype=np.float32 ,ndmin=3)
                last_expected = (last_expected - MEAN) / STD
                #print('Measured: {:.1f},{:.1f}'.format(expected[0], expected[1]))
                #print('Predicted: {:.1f},{:.1f}'.format(predicted[0, 0],predicted[0, 1]))
            temp_actual = f'{expected[0] :0.2f}'
            hum_actual = f'{expected[1] :0.2f}'
            temp_pred =  f'{predicted[0, 0] :0.2f}'
            hum_pred = f'{predicted[0, 1] :0.2f}'
            temp_abs_error = np.abs( predicted[0, 0] - expected[0] )
            hum_abs_error = np.abs(predicted[0, 1] - expected[1] )

            
            timestamp  = int((datetime.now()).timestamp()) # (05/12/2022 19:15:01)
            # check the temp ALERT condition 
            if(temp_abs_error > tthres):
                # print("*" * 100)
                # print(f" temp actual {temp_actual} {type(temp_actual)}pred {type(temp_pred)}")
                # pack message into SENML+JSON STRING
                message = {
                    'bn': 'raspberrypi.local',
                    'bt': timestamp,
                    'e':[
                        {'n': 'temperature_predicted', 
                            'u': '°C', 
                            't': 0, 
                            'v': float(predicted[0,0])},
                            {'n': 'temperature_actual', 
                            'u': '°C', 
                            't': 0, 
                            'v': float(expected[0])}  
                    ]
                }
                message = json.dumps(message)
                test.myMqttClient.myPublish("/s289815/temperature_alert", message)                    # publishing temperature alert 
                
            # check the humidity ALERT condition 
            if(hum_abs_error>hthres):
                # print("&" * 100)
                # print(f" hum actual {hum_actual} pred {hum_pred}")
                # pack message into SENML+JSON STRING
                message = {
                    'bn': 'raspberrypi.local',
                    'bt': timestamp,
                    'e':[
                        {'n': 'humidity_predicted', 
                            'u': '%', 
                            't': 0, 
                            'v': float(predicted[0,1])},
                            {'n': 'humidity_actual', 
                            'u': '%', 
                            't': 0, 
                            'v': float(expected[1])}   
                    ]
                }
                message = json.dumps(message)
                test.myMqttClient.myPublish("/s289815/humidity_alert", message)            # publishing humidity alert 

            print("*" * 100)
            
            time.sleep(1)
              

              

                

if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(ADD(), '/add', conf)
    cherrypy.tree.mount(LIST(), '/list', conf)
    cherrypy.tree.mount(PREDICT(), '/predict', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()

    cherrypy.engine.block()

    while True:
        time.sleep(1)
