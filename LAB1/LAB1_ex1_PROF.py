import adafruit_dht
import argparse
import datetime
import time
from board import D4

parser=argparse.ArgumentParser()
parser.add_argument('-f', type=float, help='frequency in seconds')
parser.add_argument('-p', type=float, help='period in seconds')
parser.add_argument('-o', type=str, help='output filename')

args = parser.parse_args()

num_samples = int(args.p // args.f)

dht_device= adafruit_dht.DHT11(D4)

fp=open(args.o, 'w')

for i in range(num_samples):
	
	now=datetime.datetime.now()
	temperature= dht_device.temperature
	humidity= dht_device.humidity
	
	#print('{:02}/{:02}/{:04},{:02}:{:02}:{:02},{:},{:}'.format(now.day, now.month, now.year, now.hour, now.minute, now.second, temperature, humidity), file=fp)
	time.sleep(args.f)
	
fp.close()


'''
python3 lab1_ex1.py -f 5 -p 20 -o ht.txt
cat ht.txt
'''
