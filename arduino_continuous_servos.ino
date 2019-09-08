#include <Servo.h> 

int rightVal = 0;          
int leftVal = 0;  

Servo rightServo; 
Servo leftServo; 

void setup()
{
  Serial.begin(9600); 
  Serial.println("Speed 0 to 90 (but reversed)");
  rightServo.attach(3);
  leftServo.attach(4);
  leftServo.write(90);
  rightServo.write(90);
}


void loop()
{
  if (Serial.available())
  {
    int speed = Serial.parseInt();
    Serial.println("Speed "+String(speed));
    //speed will be 0 (high) to 90 (low)    
    //both will do the same thing
    if(speed > 0){
      int leftServoVal  =  180 - speed;
      leftServo.write(leftServoVal);

      int rightServoVal  =  speed;
      rightServo.write(rightServoVal);

      Serial.print("rightServoVal ");
      Serial.println(rightServoVal);
      Serial.print("leftServoVal ");
      Serial.println(leftServoVal);
    
      delay(200);
    }
  }
}
