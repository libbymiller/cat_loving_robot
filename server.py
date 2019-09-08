#!/usr/bin/python

# simple server to send serial commands to an arduino on a pi

# pip install bottle

from bottle import run, route, request, response
import subprocess
import serial
import sys
import time
import os


s = "/dev/serial0"

ser = serial.Serial(s, 9600)

@route('/', method=['OPTIONS', 'POST'])
def index():

    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

    command = "none"

    if request.method == 'POST':
      command = request.POST['command']
      print(command)
      arduino_command = None
      if(command):

        print(command+"\n")
        ser.write(command.encode()+"\n")


    response.set_header('Access-Control-Allow-Origin', '*')
    result = "ok: "+command
    return result

run(host='localhost', port=8080, reloader=True)
