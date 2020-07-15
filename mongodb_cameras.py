from pymongo import MongoClient
import os 
democlient = MongoClient()
myclient = MongoClient(os.environ.get("host_db") or "localhost" , 27017)
mydb = myclient["enjoywork"]
mycol = mydb["camera_python"]
#db = democlient.appdb


def create_camera(protocol,name,direction,ip=None,url_rtsp=None):
    new_list = [{"protocol":protocol, "name": name, "direction": direction, "ip": ip, "url_rtsp":url_rtsp}]
    mycol.insert_many(new_list)
def get_camera(name):
    query = {"direction": name }
    mydoc = mycol.find(query)
    result = None
    for x in mydoc:
        result = x['ip']
    print(result)
    return result
def edit_camera(protocol,name,direction,ip=None,url_rtsp=None):
    query = {"name": name }
    new_list = {"protocol":protocol, "name": name, "direction": direction, "ip": ip, "url_rtsp":url_rtsp}
    newvalues = { "$set": new_list }
    print('okeeeeeeeeeeeeeee')
    mycol.update_one(query, newvalues,upsert=False)
def delete_camera(name):
    query = {"name": name }
    mycol.delete_one(query)
