import pickle
import os
import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
# from MongoWrapper import MongoWrapper as mdb

democlient = MongoClient()
myclient = MongoClient(os.environ.get('host_db') or 'localhost', 27017)
# print(os.environ.get('DB_PORT_27017_TCP_ADDR'))

print('check myclient', myclient)
# db = myclient.appdb.enjoywork.embedded_data
# print(db)
mydb =myclient["enjoywork"]
mycol = mydb["embedded_data"]
print('check mycol',mycol)
def add_staff(embedded,staffname):
    """insert embedded of new staff to mongoDB

    Arguments:
        embedded {pickle} -- numpy2pickle
        staffname {str} -- name of users 
    """

    # mycol.insert({"name": staffname, "embedded": pickle.dumps(embedded, protocol=2)})
    mycol.insert({"name": staffname, "embedded": pickle.dumps(embedded, protocol=2)})
    
def get_embedded(username):
    """load embedded from mongoDB to client

    Arguments:
        staff {str} -- name of staff

    Returns:
        pickle -- embedded face of staff
    """

    reconstructed_obj = np.array([])
    count = 0
    print('check error in mongo_embedded ')

    for x in mycol.find({"name":username},{"_id":0, "embedded": 1 }):

        count += 1
        embedded = x.get("embedded")
        saved_str = repr(embedded)     
        reconstructed_pickle = eval(saved_str)
        # reconstructed_obj = pickle.loads(reconstructed_pickle)
        reconstructed_obj = np.append(reconstructed_obj, pickle.loads(reconstructed_pickle), axis=None)
        if count == 1:
            break
    return reconstructed_obj, count

def get_full_embedded():
    """get full embedded to train KNN

    Returns:
        pickle  -- pickle file which is include all embe
    """


    reconstructed_obj = np.array([])
    username = []
    count = 0
    for x in mycol.find({},{"_id": 0, "name":1, "embedded":1}):
        count+=1
        embedded = x.get("embedded")
        sth = x.get("name")
        saved_str = repr(embedded)     
        reconstructed_pickle = eval(saved_str)
        reconstructed_em = pickle.loads(reconstructed_pickle)
        reconstructed_em = np.asarray(reconstructed_em).reshape(-1,128)
        print('countttt',reconstructed_em.shape[0])
        reconstructed_obj = np.append(reconstructed_obj, reconstructed_em, axis=None)
        for _ in range (reconstructed_em.shape[0]):
            username.append(sth)
        print('count',count)

    reconstructed_obj = reconstructed_obj.reshape(-1,128)
    print(username ,reconstructed_obj)

    return username, reconstructed_obj
print('check get_embedding : ',get_embedded('AnNH'))

