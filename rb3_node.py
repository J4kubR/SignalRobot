import os
import rclpy    
import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile

detected_key = 0

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

detected_key = None

SAMPLE_RATE = 44100
FRAMES_PER_BUFFER = 1024
COMMON_THRESHOLD = 10

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_detected_tones(magnitudes):
    detected_tones = []
    for freq, mag in magnitudes.items():
        if mag > COMMON_THRESHOLD:
            detected_tones.append(freq)
    return detected_tones



def callback(in_data, frame_count, time_info, status):
    indata = np.frombuffer(in_data, dtype=np.float32)
    volume_norm = np.linalg.norm(indata) * 10
    window = np.hanning(FRAMES_PER_BUFFER)

    filtered_data = {
        "697Hz": butter_bandpass_filter(indata, 690, 705, SAMPLE_RATE),
        "770Hz": butter_bandpass_filter(indata, 745, 795, SAMPLE_RATE),
        "1336Hz": butter_bandpass_filter(indata, 1310, 1365, SAMPLE_RATE),
        "1209Hz": butter_bandpass_filter(indata, 1185, 1235, SAMPLE_RATE),
        "1477Hz": butter_bandpass_filter(indata, 1450, 1505, SAMPLE_RATE),
        "852Hz": butter_bandpass_filter(indata, 830, 900, SAMPLE_RATE)
    }

    if volume_norm > 200:
        fourier = np.fft.rfft(indata * window, n=FRAMES_PER_BUFFER)
        freqs = np.fft.rfftfreq(FRAMES_PER_BUFFER, 1. / SAMPLE_RATE)

        indices = {freq: np.argmin(np.abs(freqs - int(freq.replace("Hz", "")))) for freq in filtered_data.keys()}
        magnitudes = {freq: np.abs(fourier[idx]) for freq, idx in indices.items()}

        detected = get_detected_tones(magnitudes)
        global detected_key

        if "697Hz" in detected and "1336Hz" in detected:
            detected_key = 1
        elif "770Hz" in detected and "1209Hz" in detected:
            detected_key = 2
        elif "770Hz" in detected and "1336Hz" in detected:
            detected_key = 3
        elif "770Hz" in detected and "1477Hz" in detected:
            detected_key = 4
        elif "852Hz" in detected and "1336Hz" in detected:
            detected_key = 5
        else:
            detected_key = 6

    return (in_data, pyaudio.paContinue)


def detect_dtmf_tone():
    global detected_key
    detected_key = 0

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER,
                    stream_callback=callback)

    stream.start_stream()

    while True:
        if detected_key != 0:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    return detected_key


def main():
    rclpy.init()

    qos = QoSProfile(depth=10)
    node = rclpy.create_node('rb3_node')
    pub = node.create_publisher(Twist, 'cmd_vel', qos)

    try:
        target_angular_velocity = 0.0
        target_linear_velocity = 0.0
        
        while True:

            DTMFtone = detect_dtmf_tone()

            if DTMFtone == 1:
                target_linear_velocity = 0.2
            elif DTMFtone == 2:
                target_angular_velocity = -0.9
            elif DTMFtone == 3:
                target_linear_velocity = -0.2
            elif DTMFtone == 4:
                target_angular_velocity = 0.9
            elif DTMFtone == 5:
                target_angular_velocity = 0.0
                target_linear_velocity = 0.0
            elif DTMFtone == 6:
                target_angular_velocity = 0.0
                target_linear_velocity = 0.0

            twist = Twist()

            twist.linear.x = target_linear_velocity
            twist.linear.y = 0.0
            twist.linear.z = 0.0

            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = target_angular_velocity

            pub.publish(twist)

    finally:
        twist = Twist()

        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        pub.publish(twist)
        



if __name__ == '__main__':
    main()
