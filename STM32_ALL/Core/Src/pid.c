#include "pid.h"

//定义结构体变量
PID_TypeDef pidMo1Speed;

void PID_Init(void){
	pidMo1Speed.Kp = 0;
	pidMo1Speed.Ki = 0;
	pidMo1Speed.Kd = 0;
	pidMo1Speed.Setpoint = 0.00;
	pidMo1Speed.CurrentValue = 0.0;
	pidMo1Speed.LastError = 0.0;
	pidMo1Speed.Integral = 0.0;
	pidMo1Speed.Err = 0.0;
}

float P_Calculate(PID_TypeDef *pid, float CurrentValue){
	pid->CurrentValue = CurrentValue;    //传输真实值
	pid->Err = pid->Setpoint - pid->CurrentValue;	//误差值=目标-实际
	pid->CurrentValue = pid->Kp * pid->Err;	//比例控制调节
	return pid->CurrentValue;
}

float PI_Calculate(PID_TypeDef *pid, float CurrentValue){
	pid->CurrentValue = CurrentValue;    //传输真实值
	pid->Err = pid->Setpoint - pid->CurrentValue;	//误差值=目标-实际
	pid->Integral += pid->Err;	//累计误差计算
	pid->CurrentValue = pid->Kp * pid->Err + pid->Ki * pid->Integral;	//PI控制
	return pid -> CurrentValue;
}

float PID_Calculate(PID_TypeDef *pid, float CurrentValue){
	pid->CurrentValue = CurrentValue;    //传输真实值
	pid->Err = pid->Setpoint - pid -> CurrentValue;	//误差值=目标-实际
	pid->Integral += pid->Err;	//累计误差计算
	pid->CurrentValue = pid->Kp * pid->Err + pid->Ki * pid->Integral + pid->Kd * (pid->Err - pid->LastError);	//PID控制
	pid->LastError = pid->Err;	//保存误差
	return pid -> CurrentValue;
}
