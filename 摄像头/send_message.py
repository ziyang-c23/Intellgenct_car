import serial
import time

def open_serial(port='COM6', baudrate=9600, timeout=1):
    """打开串口"""
    return serial.Serial(port, baudrate, timeout=timeout)

def send_message(ser, message):
    """发送消息到串口"""
    ser.write((message + '\n').encode())

def receive_message(ser):
    """接收串口消息"""
    rx = ser.readline().decode(errors='ignore').strip()
    return rx

if __name__ == '__main__':
    ser = open_serial()
    print('蓝牙串口已打开，输入 q 退出')
    while True:
        tx = input('>>> ')
        if tx == 'q':
            break
        send_message(ser, tx)
        rx = receive_message(ser)
        if rx:
            print('收到:', rx)
    ser.close()