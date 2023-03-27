#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>
#include <micro_ros_arduino.h>
#include <Wire.h>
#include "arduinoFFT.h"

#include "MPU9250.h"
#include "model.h"


#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

#include <std_msgs/msg/string.h>


arduinoFFT FFT = arduinoFFT(); /* Create FFT object */

#define NUMBER_OF_INPUTS 1536   // Input shape of the model trained 
#define NUMBER_OF_OUTPUTS 2     // Output shape of the model trained 
#define TENSOR_ARENA_SIZE 8*1024

#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}


#define SCL_INDEX 0x00
#define SCL_TIME 0x01
#define SCL_FREQUENCY 0x02
#define SCL_PLOT 0x03
#define LED_PIN 13

rcl_publisher_t publisher;
std_msgs__msg__String msg;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;


MPU9250 mpu;
Eloquent::TinyML::TensorFlow::TensorFlow<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> nn;

float X_test[1536] = {0} ;

   
const int numSamples = 288;                     // Number of values which we need to read before starting the inference 
const float accelerationThreshold = 2.5;          
int samplesRead = numSamples;
constexpr int MAX_MEASUREMENTS = 288 ;
float measurements[6][MAX_MEASUREMENTS];


const uint16_t samples = 64; //This value MUST ALWAYS be a power of 2




void error_loop(){
  while(1){
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(100);
  }
}

void setup() {
    set_microros_transports();
    Serial.begin(115200);
    nn.begin(model);
    Wire.begin();
    delay(2000);

                                       //////// initialize the IMU  ////////
 if (!mpu.setup(0x68)) {  
        while (1) {
           // Serial.println("MPU connection failed. Please check your connection with `connection_check` example.");
            delay(5000);
        }
  }

                                       //////// calibrate IMU ////////
                                       

    Serial.println("Please leave the device still on the flat plane.");
    mpu.verbose(true);
    delay(5000);
    mpu.calibrateAccelGyro();
    mpu.verbose(false); 
    Serial.println("Calibration DONE");

                                        //////// END of IMU calibration ////////



                                        ////////  ROS initialization  ////////
                                        
    allocator = rcl_get_default_allocator();

    //create init_options
    RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));

    // create node
    RCCHECK(rclc_node_init_default(&node, "imu_publisher_node", "", &support));
 
    // create publisher
    RCCHECK(rclc_publisher_init_default(
    &publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
    "imu_info_topic"));

                                         ////////  END  of ROS initialization  ////////
}

void loop() {
   
    float aX, aY, aZ, gX, gY, gZ;
   
  // wait for significant motion
   while (samplesRead == numSamples) {
    if (mpu.update()) { 
      // read the acceleration data
       aX = mpu.getAccX() ;
       aY = mpu.getAccY() ;
       aZ = mpu.getAccZ() ;
      
     // sum up the absolutes
       float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
             
    // check if it's above the threshold
     if (aSum >= accelerationThreshold) {
     // reset the sample read count
      samplesRead = 0;
      
      break;
               }
     }
  }

  
  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  int NewStart = 0 ;
  int NewEnd = 32 ;
  int cnt = 0 ;
  while (samplesRead < numSamples)
   {   
      mpu.update() ;
      // read the acceleration and gyroscope data
      aX = mpu.getAccX() ;
      aY = mpu.getAccY() ;
      aZ = mpu.getAccZ() ;
      gX =  mpu.getGyroX() ;
      gY =  mpu.getGyroY() ;
      gZ =  mpu.getGyroZ() ;
      
      // normalize the IMU data between 0 to 1 and store it
      
      measurements[0][cnt] = aX ;
      measurements[1][cnt] = aY ;
      measurements[2][cnt] = aZ ;
      measurements[3][cnt] = gX ;
      measurements[4][cnt] = gY ;
      measurements[5][cnt] = gZ ;
      cnt++ ;
      
      delay(1);

      samplesRead++;     
    }   

    for (int axis = 0; axis < 6; axis++) 
    {
      int counter = 0 ; 
      for ( int i = 0; i < 8; i++)//8
      {
          double vReal[64] = {0} ;
          double vImag[64] = {0};
          for ( int j = 0; j < 64; j++)
          {
            vReal[j] = measurements[axis][j+counter] ; 
          }
          
          FFT.Windowing(vReal, samples, FFT_WIN_TYP_HANN, FFT_FORWARD);  
          
          FFT.Compute(vReal, vImag, samples, FFT_FORWARD); 
         
          FFT.ComplexToMagnitude(vReal, vImag, samples); 
          
         
          int aug = 0 ;
          for ( int j = NewStart; j < NewEnd; j++)
          {   
            X_test[j] = vReal[aug] ;
            aug++ ;
          }
          
          NewStart += 32 ;
          NewEnd += 32 ;  
          counter = counter + 32 ;
      }
      
    }
 
      // Test if the required samples have been attended
      if ( samplesRead == numSamples ) {
      if(nn.predictClass(X_test) == 0 )
      {  
        msg.data.data = "RIGHT" ;
        RCSOFTCHECK(rcl_publish(&publisher, &msg, NULL));
      }
      else if(nn.predictClass(X_test) == 1 )
      { 
        msg.data.data = "UP" ;
        RCSOFTCHECK(rcl_publish(&publisher, &msg, NULL));
      }
      
     }
    
}





 
