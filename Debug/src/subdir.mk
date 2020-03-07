
# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/harrisCornerDetect.cpp \
../src/implementHarris.cpp 

OBJS += \
./src/harrisCornerDetect.o \
./src/implementHarris.o 

CPP_DEPS += \
./src/harrisCornerDetect.d \
./src/implementHarris.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++-7 -I/usr/local/include/opencv4 -O0 -g3 -Wall -c -fmessage-length=0 -std=c++17 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


