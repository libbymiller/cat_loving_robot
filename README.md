# cat_loving_robot

A robot for danbri, using Raspberry Pi, arduino and continuous rotation servos. It races towards cats when it sees them.

Details: https://planb.nicecupoftea.org/2018/12/01/cat-detector-with-tensorflow-on-a-raspberry-pi-3b/

Note: the server requires python2 and the tensorflow part python3!

The arduino part is for a couple of continuous rotation servos. I assume the serial is connected via the GPIO:

5v on Pi to 5v on the arduino (e.g. GPIO physical numbering 2 or 4)
ground on the Pi to ground on the Arduino (e.g. GPIO physical numbering 6, 9 etc)
Pi's TDX (GPIO physical numbering pin 8) to Arduino's RX

Normally you'd need a voltage divider to do this serial communication, but because we're only going one way - from the Pi to 
the Arduino - we get away with it.


