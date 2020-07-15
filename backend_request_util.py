'''
This is a util to communicate to http request of Dsoft Backend server
Developed by triqn@d-soft.com.vn
'''
import os
import requests
import api_constant as api
from pyfcm import FCMNotification

IS_LOGINED = False
TOKEN = ''

class DsoftHttpRequest():
    """
    Initial the Http Request to backend server
    Set the username and password of caster user 
    Initial the Firebase Notification
    """
    def __init__(self):
        self.username = api.API_USER
        self.password = api.API_PWD
        self.token = ''
        self.push_fcm = FCMNotification(api_key=api.FCM_KEY)
    """
    Login to Backend Server with user info from api constant file.
    Save the token to memory variable
    """
    def login(self):
        jsonData = {"username": self.username, "password": self.password}
        r = requests.post(api.LOGIN_API, json=jsonData)
        if r.status_code == 200:
            self.token = r.json()['data']['access_token']
            print('Login successful with access-token: {}'.format(self.token))
        else:
            self.token = ''
            print('Login failed with error: {}'.format(str(r.status_code)))
    """
    Get token
    """    
    def __get_token(self):
        return self.token
    """
    Post work log when user is recognited
    @param: image_path = Image path is set into Backend Server /publish/evidence
    @param: username = Name of staff is recognited
    @param: status = Direction In or Out
    """
    def post_worklog(self, image_path, username, status):
        userId = self.get_user_id(username)
        
        if userId == None:
            print('Failed to get username')
            return
        data = {"image": image_path}
        url = api.POST_WORKLOG + '/{}/{}'.format(userId, status) 
        print(url)
        print(userId)
        r = requests.post(url, json=data, headers={'access-token': self.token})
        if r.status_code == 200:
            print("Post worklog successful!")
        else:
            print("Failed to post worklog!")
        return r.status_code
        #self.__post_method(url, files= data)
    # def post_worklog(self, image_path, username, status):
    #     userId = self.get_user_id(username)
        
    #     if userId == None:
    #         print('Failed to get username')
    #         return
    #     data = {"_id": userId, "username": username, "status": status, "image": image_path}
    #     url = api.POST_WORKLOG + '/{}/{}'.format(userId, status) 
        
    #     print(userId)
    #     r = requests.post(url, json=data, headers={'access-token': self.token})
    #     if r.status_code == 200:
    #         print("Post worklog successful!")
    #     else:
    #         print("Failed to post worklog!")
        #self.__post_method(url, files= data)
    """
    Get user info to return to client
    @param: user_name = Username of staff
    @return: userInformation
    """
    def get_user_info(self, user_name):
        userID = self.get_user_id(user_name)
        url = api.GET_USER_INFO.format(userID)
        data = self.__get_method(url)
        return data
    """
    Get user ID
    @param: username = Username of staff need to get UserID
    @return userID = String of userID
    """
    def get_user_id(self, username):
        username = username.lower()
        url = api.GET_USER_ID.format(username)
        return self.__get_method(url)
    """
    Generic GET method
    @param: url: Url of get method
    @return:
        200: Successful
        403: Error Not Found
    """
    def __get_method(self, url):
        if not self.token:
            print("Token empty, try to login again!")
            self.login()
        r = requests.get(url, headers={'access-token': self.token})
        if r.status_code == 200:
            return r.json()["data"]
        elif r.status_code == 403:
            print(r.json()["errors"])
            # Try to login again
            self.login()
            return
    def opendoor(self, url, myfiles):
        r = requests.post(url, json = myfiles)
        return r.text


    """
    Generic POST method
    @param: url = URL of post method
    @param: files = Files to upload if need
    """
    def __post_method(self, url, files = None):
        if not self.token:
            print("Token empty, try to login again!")
            self.login()
        r = requests.post(url, files= files, headers={'access-token': self.token})
        print(r.status_code)
    """
    Send the notification with firebase
    @param: image_src: Path of image evidence
    """
    def notify_stranger(self, image_src):
        self.push_fcm.notify_topic_subscribers(
            topic_name = 'unknown',
            message_title = 'Stranger detect!!!',
            message_body = image_src
        )
        return 200


