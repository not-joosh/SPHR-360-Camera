// #include "StepperMotor.h"

// StepperMotor::StepperMotor(int pin1, int pin2, int pin3, int pin4): pin1_(pin1), pin2_(pin2), pin3_(pin3), pin4_(pin4) {
//     pinMode(pin1_, OUTPUT);
//     pinMode(pin2_, OUTPUT);
//     pinMode(pin3_, OUTPUT);
//     pinMode(pin4_, OUTPUT);
// }

// // DIRECTION = TRUE -> CLOCKWISE/RIGHT
// // DIRECTION = FALSE -> COUNTERCLOCKWISE/LEFT
// void StepperMotor::setAngle(int angle, bool direction) {
//     int steps = angle/0.088/4/2;

//     for (int i = 0; i < steps; i++) {
//         if (direction) {
//             for (int j = 0; j < 4; j++) {
//                 digitalWrite(pinArray_[j], (int) (i/4)%2 == j);
//             }
//         } else {
//             for (int j = 3; j >= 0; j--) {
//                 digitalWrite(pinArray_[j], (int) (i/4)%2 == j);
//             }
//         }
//         delay(2);
//     }
// }