# Flask constance

AI_FLASK_HOST = '192.168.4.24'

AI_FLASK_PORT = 5080

AI_TRIGGER_START_FLASK = 'http://' + AI_FLASK_HOST + ':' + str(AI_FLASK_PORT) + '/started'

# Server constance
API_USER = 'dsoft'

API_PWD = 'dsoft@1607'

HTTP_API_URL = 'http://192.168.4.24:9000'

LOGIN_API = HTTP_API_URL + '/auth/login'

POST_WORKLOG = HTTP_API_URL + '/dashboard'

GET_WORKLOG = HTTP_API_URL + '/dashboard'

OPEN_DOOR = 'http://192.168.4.27:5080/open'

WELCOME = HTTP_API_URL + '/users/{}/getfullname'

UNKNOW_MESSAGE = HTTP_API_URL + '/users/unknown'

GET_USER_ID = HTTP_API_URL + '/users/{}/getid'

SPEAKER = 'http://192.168.3.7:8100/play/'

FCM_KEY = "AAAAUbp_7eg:APA91bHtpg6wTT1ez7gal7DV17pk5I7aNl7FvFcMWNIMU7iLwou_UF9Waz3Qs4rrdf3phZggyxQjB8Kuumwk47Ec6ecEy3hcxwxnXOor68CL2J3RS04udCw0In_Lg1IHZ7H0937ivb9X"

RECORD_PERSON_PATH = '/home/dsoft/Projects/FaceLog/FaceLog.Server/server/public/records'
#rtsp://dsoft:Dsoft@321@113.176.195.116:554/ch1/main/av_stream

#rtsp://dsoft:Dsoft@321@113.176.195.116:555/ch1/main/av_stream
# PI_CAM1 = "rtsp://dsoft:Dsoft@321@113.176.195.116:555/ch1/main/av_stream"
# PI_CAM2 = "rtsp://dsoft:Dsoft@321@113.176.195.116:555/ch1/main/av_stream"
#Camera gstreamer
#PI_CAMERA_1 = 'udpsrc port=5200 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=JPEG, payload=(int)26" ! rtpjpegdepay ! decodebin ! videoconvert ! appsink'
#PI_CAMERA_1 = 'http://192.168.4.4:5080/video_out.mjpg'
PI_CAMERA_1 = 'http://192.168.3.200:5080/video_stream.mjpg'
#PI_CAMERA_2 = 'udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=JPEG, payload=(int)26" ! rtpjpegdepay ! decodebin ! videoconvert ! appsink'
PI_CAMERA_2 = 'http://192.168.3.205:5080/video_stream.mjpg'
CAMERA_TOP = "rtsp://dsoft:Dsoft@321@113.176.195.116:555/ch1/main/av_stream"
# CAMERA_TOP = 'rtsp://admin:dsoft@1607@192.168.3.200/ch1/sub_stream/av_stream'
# Visualization color
box_conner = (255,255,255) # White

unknown_color = (0, 255,  255) # Yellow

recog_color = (255, 0, 0) # Blue

# Extension image allows
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])