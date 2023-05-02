import uwebsockets.client
import sys,time
import network
from machine import Pin
from time import sleep
import _thread
import upip

import machine, time
from machine import Timer

TIMEOUT_MS = 5000 #soft-reset will happen around 5 sec

def timeout_callback(t):
    try:
        print("I am in Call back")
        websocket.send("Ws send \r")
    finally:
        print("I am in Call back finally")
       

def wlan_connect(ssid,pwd):
    wlan = network.WLAN(network.STA_IF)
    if not wlan.active() or not wlan.isconnected():
        wlan.active(True)
        wlan.connect(ssid,pwd)
        while not wlan.isconnected():
            pass
    print('network config:', wlan.ifconfig())
    #upip.install("")
    
    
wlan_connect('iPhone', '123454321')
p0 = Pin(2, Pin.OUT) 
uri = "wss://gtg3p8yh66.execute-api.us-east-1.amazonaws.com/production/"
websocket = uwebsockets.client.connect(uri)
#print(websocket)
while True:
    #print("Enter Command:\r")
    #mesg=input()
    #websocket.send(mesg + "\r")
    timer = Timer(0)
    timer.init(period=TIMEOUT_MS, callback=timeout_callback)
    resp = websocket.recv()
    if "on" in resp:
        p0.on()
        print("Led is on")
    elif "off" in resp:
        p0.off()
        print("Led is off")
    else:
        print("No data recieved")
    sleep(1)

