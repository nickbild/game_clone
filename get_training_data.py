import os
import time
import keyboard


# 'xwininfo' command retrieves window ID.
window_id = "0x220000a"

print("Get ready...")
for i in range(10, -1, -1):
    print(i)
    time.sleep(1)
print("Go!")

for _ in range(1000):
    cur_time = time.time()

    for cnt in range(6):
        # Get state of inputs.
        state = 0
        if keyboard.is_pressed('up'):
            state += 1
        if keyboard.is_pressed('down'):
            state += 2
        if keyboard.is_pressed('w'):
            state += 4
        if keyboard.is_pressed('s'):
            state += 8
        
        # Capture a screenshot.
        os.system("convert x:{0} img/screen-{1}-{2}-{3}.jpg".format(window_id, cur_time, cnt, str(state)))

        time.sleep(0.05)
