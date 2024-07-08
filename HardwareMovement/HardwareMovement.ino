#include <Servo.h>

#define SERVO_PIN 3
#define MOTOR_IN1 7
#define MOTOR_IN2 8
#define MOTOR_IN3 9
#define MOTOR_IN4 10

Servo servo;
int pinArray[] = {MOTOR_IN1, MOTOR_IN2, MOTOR_IN3, MOTOR_IN4};

int servoPos = 0;

void setup() {

  // Sets up the servo object and connects the signal pin to pin 3 on the Arduino
  servo.attach(SERVO_PIN);

  // Sets up the motor pins as outputs
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(MOTOR_IN3, OUTPUT);
  pinMode(MOTOR_IN4, OUTPUT);
  
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    int commaIndex = data.indexOf(',');
    if (commaIndex != -1) {
      String offsetXString = data.substring(0, commaIndex);
      String offsetYString = data.substring(commaIndex + 1);
      int offset_x = offsetXString.toInt();
      int offset_y = offsetYString.toInt();
      
      int tmp = offset_x/50;
      int tmp2 = offset_y/90;

      if (tmp > 0) {
        setMotorAngle(abs(tmp), true);
      } else {
        setMotorAngle(abs(tmp), false);
      }

      if (tmp2 < 1 && tmp2 > -1){}
      else{
        servoPos += tmp2;

        if (servoPos <= 180 && servoPos >= 0)
          servo.write(servoPos);
        else
          servoPos = 90;
      }
    }
  }
}

void setMotorAngle(int angle, bool direction){
  int steps = angle/0.088/4/2;

  for (int i = 0; i < steps; i++) {

    if (direction) {

      for (int j = 0; j < 4; j++){
        digitalWrite(pinArray[j], HIGH);
        delay(2);
        digitalWrite(pinArray[j], LOW);
      }

    } else {

      for (int j = 3; j >= 0; j--) {

        digitalWrite(pinArray[j], HIGH);
        delay(2);
        digitalWrite(pinArray[j], LOW);
      }
    }
  }
}