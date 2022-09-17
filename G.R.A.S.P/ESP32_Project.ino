#include <LiquidCrystal.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// Class Objects
LiquidCrystal lcd(32, 33, 14, 27, 26, 25);
Adafruit_MPU6050 mpu;

// Object event to pull sensor readings
sensors_event_t a, g, temp;

// Force sensor pins (analogread)
#define FORCE_SENSOR_PIN1 15  // the FSR1 and 2.4k pulldown are connected to P15
#define FORCE_SENSOR_PIN2 2  // the FSR2 and 2.4k pulldown are connected to P2
#define FORCE_SENSOR_PIN3 4  // the FSR3 and 2.4k pulldown are connected to P4
#define FORCE_SENSOR_PIN4 34  // the FSR4 and 1k pulldown are connected to P34

// LED color pins (analogwrite)
#define LED_PIN_RED 18         // the LED Red pin is connected to P5
#define LED_PIN_GREEN 17      // the LED Green pin is connected to P17
#define LED_PIN_BLUE 5       // the LED Blue pin is connected to P16

// Initialize Global Variables
int analogReading1, analogReading2, analogReading3, analogReading4;
int analog_offset1, analog_offset2, analog_offset3, analog_offset4 = 0;
float tilt_angle;

// Sensor Thresholds
int light_touch = 10;
int light_squeeze = 300;
int medium_squeeze = 1200;
int big_squeeze = 1900;

// IMU offsets (run calibration script to obtain values)
float ax_offset = 0.51;
float ay_offset = 0.2;
float az_offset = 0.75;
float gx_offset = 0.06;
float gy_offset = 0.03;
float gz_offset = 0.04;

// ----------------------------- SETUP ------------------------------- //
void setup() {
  Serial.begin(115200);

  while (!Serial){
    delay(10); // will pause until serial begins
  }
  
  Serial.println("Serial port initialized.");
  delay(2000);

  lcd.begin(16, 2);
  delay(500);

  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Calibrating");
  lcd.setCursor(0,1);
  lcd.print("Please wait");

  mpu_setup();
  force_sensor_setup();
  delay(5000);

  pinMode(LED_PIN_RED,OUTPUT);
  pinMode(LED_PIN_GREEN,OUTPUT);
  pinMode(LED_PIN_BLUE,OUTPUT);

  Serial.println("Initialization complete... Starting script.");
  lcd.clear();
  delay(500);
  lcd.setCursor(0,0);
  lcd.print("One moment");
  lcd.setCursor(0,1);
  lcd.print("please :)");
  delay(5000);

}

// ------------------------------ LOOP ------------------------------ //
void loop() { 

  /* Get new sensor events with the readings */
  mpu.getEvent(&a, &g, &temp);
  
  a.acceleration.x = a.acceleration.x - ax_offset;
  a.acceleration.y = a.acceleration.y - ay_offset;
  a.acceleration.z = a.acceleration.z - az_offset;

  tilt_angle = acos(a.acceleration.y / 9.8)*57.2958;

  g.gyro.x = g.gyro.x - gx_offset;
  g.gyro.y = g.gyro.y - gy_offset;
  g.gyro.z = g.gyro.z - gz_offset;

  analogReading1 = analogRead(FORCE_SENSOR_PIN1) - analog_offset1;
  analogReading2 = analogRead(FORCE_SENSOR_PIN2) - analog_offset2;
  analogReading3 = analogRead(FORCE_SENSOR_PIN3) - analog_offset3;
  analogReading4 = analogRead(FORCE_SENSOR_PIN4) - analog_offset4;

  /* Print out the values */
  Serial.print("Acceleration X: ");
  Serial.print(a.acceleration.x);
  Serial.print(", Y: ");
  Serial.print(a.acceleration.y);
  Serial.print(", Z: ");
  Serial.print(a.acceleration.z);
  Serial.println(" m/s^2");

  Serial.print("Rotation X: ");
  Serial.print(g.gyro.x);
  Serial.print(", Y: ");
  Serial.print(g.gyro.y);
  Serial.print(", Z: ");
  Serial.print(g.gyro.z);
  Serial.println(" rad/s");

  Serial.print("Temperature: ");
  Serial.print(temp.temperature);
  Serial.println(" degC");

  Serial.println("");

  Serial.println("Force sensor reading:");

// FORCE SENSOR 1
  Serial.print("Force Sensor 1 = ");
  Serial.print(analogReading1); // print the raw analog reading

  if (analogReading1 < light_touch)       // from 0 to 9
    Serial.println(" -> no pressure");
  else if (analogReading1 < light_squeeze) // from 10 to 199
    Serial.println(" -> light touch");
  else if (analogReading1 < medium_squeeze) // from 200 to 499
    Serial.println(" -> light squeeze");
  else if (analogReading1 < big_squeeze) // from 500 to 799
    Serial.println(" -> medium squeeze");
  else // from 800 to 1023
    Serial.println(" -> big squeeze");

// FORCE SENSOR 2
  Serial.print("Force Sensor 2 = ");
  Serial.print(analogReading2); // print the raw analog reading
  
  if (analogReading2 < light_touch)       // from 0 to 9
    Serial.println(" -> no pressure");
  else if (analogReading2 < light_squeeze) // from 10 to 199
    Serial.println(" -> light touch");
  else if (analogReading2 < medium_squeeze) // from 200 to 499
    Serial.println(" -> light squeeze");
  else if (analogReading2 < big_squeeze) // from 500 to 799
    Serial.println(" -> medium squeeze");
  else // from 800 to 1023
    Serial.println(" -> big squeeze");

// FORCE SENSOR 3
  Serial.print("Force Sensor 3 = ");
  Serial.print(analogReading3); // print the raw analog reading
  
  if (analogReading3 < light_touch)       // from 0 to 50
    Serial.println(" -> no pressure");
  else if (analogReading3 < light_squeeze) // from 10 to 199
    Serial.println(" -> light touch");
  else if (analogReading3 < medium_squeeze) // from 200 to 499
    Serial.println(" -> light squeeze");
  else if (analogReading3 < big_squeeze) // from 500 to 799
    Serial.println(" -> medium squeeze");
  else // from 800 to 1023
    Serial.println(" -> big squeeze");

// FORCE SENSOR AVG
  int avgSensorReading = (analogReading1+analogReading2+analogReading3)/3;
  Serial.print("Average Sensor= ");
  Serial.print(avgSensorReading); // print the raw analog reading
  
  if (avgSensorReading < light_touch){      // from 0 to 50
    Serial.println(" -> no pressure");
    digitalWrite(LED_PIN_RED, LOW);
    digitalWrite(LED_PIN_GREEN, LOW);
    digitalWrite(LED_PIN_BLUE, LOW);
  }
  else if (avgSensorReading < light_squeeze){ // from 10 to 199
    Serial.println(" -> light touch");
    digitalWrite(LED_PIN_RED, HIGH);
    digitalWrite(LED_PIN_GREEN, HIGH);
    digitalWrite(LED_PIN_BLUE, HIGH);
  }
  else if (avgSensorReading < medium_squeeze){ // from 200 to 499
    Serial.println(" -> light squeeze");
    digitalWrite(LED_PIN_RED, LOW);
    digitalWrite(LED_PIN_GREEN, LOW);
    digitalWrite(LED_PIN_BLUE, HIGH);
  }
  else if (avgSensorReading < big_squeeze){ // from 500 to 799
    Serial.println(" -> medium squeeze");
    digitalWrite(LED_PIN_RED, LOW);
    digitalWrite(LED_PIN_GREEN, HIGH);
    digitalWrite(LED_PIN_BLUE, LOW);
  }
  else{ // from 800 to 1023
    Serial.println(" -> big squeeze");
    digitalWrite(LED_PIN_RED, HIGH);
    digitalWrite(LED_PIN_GREEN, LOW);
    digitalWrite(LED_PIN_BLUE, LOW);
  }

  // FORCE SENSOR 4
  Serial.print("Pen-Paper Force Sensor= ");
  Serial.println(analogReading4); // print the raw analog reading

//  if (analogReading4 < light_touch)       // from 0 to 9
//    Serial.println(" -> no pressure");
//  else if (analogReading4 < light_squeeze) // from 10 to 199
//    Serial.println(" -> light touch");
//  else if (analogReading4 < medium_squeeze) // from 200 to 499
//    Serial.println(" -> light squeeze");
//  else if (analogReading4 < big_squeeze) // from 500 to 799
//    Serial.println(" -> medium squeeze");
//  else // from 800 to 1023
//    Serial.println(" -> big squeeze");
  
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Total Force:");
  lcd.setCursor(12,0);
  lcd.print(avgSensorReading);

  lcd.setCursor(0,1);  lcd.print("Tilt Angle:");
  lcd.setCursor(11,1); lcd.print(tilt_angle);
  

  delay(1000);
}

// ------------------------------ MPU ------------------------------- //
void mpu_setup(){
  Serial.println("Adafruit MPU6050 Setup");

  // Try to initialize!
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
  case MPU6050_RANGE_2_G:
    Serial.println("+-2G");
    break;
  case MPU6050_RANGE_4_G:
    Serial.println("+-4G");
    break;
  case MPU6050_RANGE_8_G:
    Serial.println("+-8G");
    break;
  case MPU6050_RANGE_16_G:
    Serial.println("+-16G");
    break;
  }
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
  case MPU6050_RANGE_250_DEG:
    Serial.println("+- 250 deg/s");
    break;
  case MPU6050_RANGE_500_DEG:
    Serial.println("+- 500 deg/s");
    break;
  case MPU6050_RANGE_1000_DEG:
    Serial.println("+- 1000 deg/s");
    break;
  case MPU6050_RANGE_2000_DEG:
    Serial.println("+- 2000 deg/s");
    break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
  Serial.print("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
  case MPU6050_BAND_260_HZ:
    Serial.println("260 Hz");
    break;
  case MPU6050_BAND_184_HZ:
    Serial.println("184 Hz");
    break;
  case MPU6050_BAND_94_HZ:
    Serial.println("94 Hz");
    break;
  case MPU6050_BAND_44_HZ:
    Serial.println("44 Hz");
    break;
  case MPU6050_BAND_21_HZ:
    Serial.println("21 Hz");
    break;
  case MPU6050_BAND_10_HZ:
    Serial.println("10 Hz");
    break;
  case MPU6050_BAND_5_HZ:
    Serial.println("5 Hz");
    break;
  }
  Serial.print("Accel_X offset: ");
  Serial.println(ax_offset);
  Serial.print("Accel_Y offset: ");
  Serial.println(ay_offset);
  Serial.print("Accel_Z offset: ");
  Serial.println(az_offset);
}

// ----------------------------- FORCE SENSOR ------------------------------ //
void force_sensor_setup(){
  Serial.println("Initializing Sensor Readings");
  

  int tracker_1, tracker_2, tracker_3 = 0;
  
  while(1){
    if(tracker_1 >= 3){
      tracker_1 = 0;
      lcd.setCursor(11,1); lcd.print("   ");
      lcd.setCursor(11,1);
    }
    if(tracker_2 >= 100){
      tracker_2 = 0;
      tracker_1++;
      lcd.print(".");
    }
    delay(5);
    
    analogReading1 = analogRead(FORCE_SENSOR_PIN1);
    analogReading2 = analogRead(FORCE_SENSOR_PIN2);
    analogReading3 = analogRead(FORCE_SENSOR_PIN3);
    analogReading4 = analogRead(FORCE_SENSOR_PIN4);
    analog_offset1 = analog_offset1 + analogReading1;
    analog_offset2 = analog_offset2 + analogReading2;
    analog_offset3 = analog_offset3 + analogReading3;
    analog_offset4 = analog_offset4 + analogReading4;
    
    tracker_2++;
    tracker_3++;

    if(tracker_3 >= 1000){
      break;
    }
    
  }
  
  analog_offset1 = analog_offset1 / tracker_3;
  analog_offset2 = analog_offset2 / tracker_3;
  analog_offset3 = analog_offset3 / tracker_3;
  analog_offset4 = analog_offset4 / tracker_3;

  lcd.clear();
  lcd.setCursor(0,0); lcd.print("Force Sensor");
  lcd.setCursor(0,1); lcd.print("Offset: ");

  delay(4000);
  
  lcd.clear();
  lcd.setCursor(0,0); lcd.print("F1: ");
  lcd.setCursor(4,0); lcd.print(analog_offset1);
  lcd.setCursor(8,0); lcd.print("F2: ");
  lcd.setCursor(12,0); lcd.print(analog_offset2);
  lcd.setCursor(0,1); lcd.print("F3: ");
  lcd.setCursor(4,1); lcd.print(analog_offset3);
  lcd.setCursor(8,1); lcd.print("F4: ");
  lcd.setCursor(12,1); lcd.print(analog_offset4);
    
  Serial.print("Force sensor 1 offset: ");
  Serial.println(analog_offset1);
  Serial.print("Force sensor 2 offset: ");
  Serial.println(analog_offset2);
  Serial.print("Force sensor 3 offset: ");
  Serial.println(analog_offset3);
  Serial.print("Force sensor 4 offset: ");
  Serial.println(analog_offset4);
   
}
