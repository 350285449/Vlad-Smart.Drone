from turtledemo.penrose import start

from djitellopy import  Tello
import cv2

width = 320
height = 240
startCounter = 1


me = Tello()
me.connect()
me.for_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0

print(me.get_battery())

me.streamoff()
me.streamon()

while True:
    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    img = cv2.resize(myFrame, (width, height))

    if startCounter == 0:
        me.takeoff()
        me.move_left(20)
        me.rotate_clockwise(90)
        startCounter = 1
    cv2.imshow("My Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break