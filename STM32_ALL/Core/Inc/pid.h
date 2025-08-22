#ifndef __PID_H
#define __PID_H

// PID控制结构体定义
typedef struct {
    float Kp;        // 比例系数
    float Ki;        // 积分系数
    float Kd;        // 微分系数
    float Setpoint; // 目标值
    float CurrentValue; // 实际值
    float LastError; // 上一次误差
    float Integral;  // 积分累积
    float Err; //误差值
} PID_TypeDef;

// PID参数初始化函数
void PID_Init(void);
// PID计算函数
float P_Calculate(PID_TypeDef *pid, float currentValue);
float PI_Calculate(PID_TypeDef *pid, float currentValue);
float PID_Calculate(PID_TypeDef *pid, float currentValue);
#endif
