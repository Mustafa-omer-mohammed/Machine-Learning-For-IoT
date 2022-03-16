import requests
import base64



# import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument('--model',default = 'cnn' , type = str ,  help='Model to be added [cnn , mlp]')
# parser.add_argument('--command',   type = str , help='Command to execute [add, list , predict]')
# parser.add_argument('--tthres', default = 0.1 , type = float , help='Temprature Threshold')
# parser.add_argument('--hthres',default = 0.2, type = float , help='Humidity Threshold')
# args = parser.parse_args()

# ######################################### argparse inputs  #########################################
# model_name = args.model 
# command = args.command
# tthres = args.tthres
# hthres = args.hthres
# print(f'model_name = {model_name}')

def main() :
    
######################################### ADD service #########################################
    path = './models'
    models = ['cnn' , 'mlp']
    for model_name in models :
        model_bytes = bytearray(open(f'{path}/{model_name}.tflite','rb').read())


        model_base64bytes =  base64.b64encode(model_bytes)
        model_string = model_base64bytes.decode()

        body = {
                "model": model_string,
                "name": model_name
                }             
        url_add = 'http://192.168.43.114:8080/add'
        r_add = requests.post(url_add, json = body)
        if r_add.status_code == 200:
            print(f" model {model_name} was added Successfully")
        else:
            print('Error:', r_add.status_code)


if __name__ == "__main__":
    main()   

