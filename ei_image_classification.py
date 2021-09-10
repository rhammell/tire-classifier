# Edge Impulse - OpenMV Image Classification Example

import sensor, image, time, os, tf, pyb
from pyb import Pin

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.GRAYSCALE) # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = "trained.tflite"
labels = [line.rstrip('\n') for line in open("labels.txt")]

# Define output pins for RGB LED
pin_r = Pin("P7", Pin.OUT_PP)
pin_g = Pin("P8", Pin.OUT_PP)
pin_b = Pin("P9", Pin.OUT_PP)

# Function to set RGB LED color based on input label
def color_by_label(label):
    if label == 'flat':
        # Red
        pin_r.high()
        pin_g.low()
        pin_b.low()
    elif label == 'full':
        # Green
        pin_r.low()
        pin_g.high()
        pin_b.low()
    elif label == 'no-tire':
        # Yellow
        pin_r.high()
        pin_g.high()
        pin_b.low()

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()

    # default settings just do one detection... change them to search the image...
    for obj in tf.classify(net, img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        img.draw_rectangle(obj.rect())
        # This combines the labels and confidence values into a list of tuples
        predictions_list = list(zip(labels, obj.output()))

        # Update RGB LED color based on the highest predicted class
        max_prediction = max(predictions_list, key=lambda item:item[1])
        color_by_label(max_prediction[0])

        for i in range(len(predictions_list)):
            #print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))
            img.draw_string(5, 10*i+5,"%s = %f" % (predictions_list[i][0].upper(), predictions_list[i][1]))

    print(clock.fps(), "fps")
