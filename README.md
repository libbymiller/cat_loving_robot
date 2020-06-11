# classify_image.py

A tensorflow imagenet model hooked up to a Raspberry Pi camera. 

See https://planb.nicecupoftea.org/2020/05/22/pi-opencv-tensorflow-again/ for details on how to use it.

# classify_image_cat.py

A robot for danbri, using Raspberry Pi, arduino and continuous rotation servos. It races towards cats when it sees them.

Details: https://planb.nicecupoftea.org/2018/12/01/cat-detector-with-tensorflow-on-a-raspberry-pi-3b/

- but use https://planb.nicecupoftea.org/2020/05/22/pi-opencv-tensorflow-again/ for updated instructions for tf and opencv for Buster.

The arduino part is for a couple of continuous rotation servos. I assume the serial is connected via the GPIO:

* 5v on Pi to 5v on the arduino (e.g. GPIO physical numbering 2 or 4)
* ground on the Pi to ground on the Arduino (e.g. GPIO physical numbering 6, 9 etc)
* Pi's TDX (GPIO physical numbering pin 8) to Arduino's RX

Normally you'd need a voltage divider to do this serial communication, but because we're only going one way - from the Pi to 
the Arduino - we get away with it.

# install

More details for the Pi here: https://planb.nicecupoftea.org/2020/05/22/pi-opencv-tensorflow-again/

For the cat-robot part, you also need: 

    pip install pyserial
    pip install bottle

# deploy

For autostart on boot you need to do this sort of thing:

    sudo cp image-detect-start.service /etc/systemd/system/
    sudo systemctl enable image-detect-start
    sudo systemctl start image-detect-start

Those `.service` systemd files assume you are using the python in /home/pi/env.

Look for logs / fails in `sudo tail -f /var/log/syslog`

