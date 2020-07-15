# Flask AI Constant
"""
The default address set to 0.0.0.0 to get the real local server address
"""
AI_FLASK_HOST = '0.0.0.0'
"""
Setting port of Flask App to 5080
"""
AI_FLASK_PORT = 5080
"""
The status url to check if the Flask server is running up
"""
AI_TRIGGER_START_FLASK = 'http://' + AI_FLASK_HOST + ':' + str(AI_FLASK_PORT) + '/started'

# Server API Constant
"""
Setting default the account Caster to login to the Backend server.
"""
API_USER = 'dsoft'
API_PWD = 'dsoft@1607'
"""
Base URL
"""
HTTP_API_URL = 'http://192.168.4.24:9000'
"""
Login URL
"""
LOGIN_API = HTTP_API_URL + '/auth/login'
"""
POST WORKLOG URL 
"""
POST_WORKLOG = HTTP_API_URL + '/dashboard'
"""
DISPLAY UNKNOWN Notification
"""
UNKNOW_MESSAGE = HTTP_API_URL + '/users/unknown'
"""
GET USER ID
@param: username
"""
GET_USER_ID = HTTP_API_URL + '/users/{}/getid'
"""
GET USER INFO
@param: userID
"""
GET_USER_INFO = HTTP_API_URL + '/users/{}'
# Firebase
"""
Firebase token
"""
FCM_KEY = "AAAAUbp_7eg:APA91bHtpg6wTT1ez7gal7DV17pk5I7aNl7FvFcMWNIMU7iLwou_UF9Waz3Qs4rrdf3phZggyxQjB8Kuumwk47Ec6ecEy3hcxwxnXOor68CL2J3RS04udCw0In_Lg1IHZ7H0937ivb9X"

#Camera Constant
# PI_CAMERA_1 = 'udpsrc port=5200 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=JPEG, payload=(int)26" ! rtpjpegdepay ! decodebin ! videoconvert ! appsink'
"""
CAMERA IN
"""
PI_CAMERA_1 = 'http://192.168.4.95:5080/video_out.mjpg'
# PI_CAMERA_2 = 'udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=JPEG, payload=(int)26" ! rtpjpegdepay ! decodebin ! videoconvert ! appsink'
"""
CAMERA OUT
"""
PI_CAMERA_2 = "http://192.168.4.26:5080/video-out.mjpg"
"""
CAMERA TOP
"""
CAMERA_TOP = 'rtsp://admin:dsoft@1607@192.168.3.200/ch1/main/av_stream'

# Extension image allows
"""
ALLOW UPLOAD IMAGE WITH EXTENSIONS
"""
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])