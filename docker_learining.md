## Detail key in docker-compose.yml 
* volumes : using when you want to share the data directory in host machines and container 

* chmod : when using volume you need to consider to set permission for folder you want to volume
* docker-compose build --force-rm : create and rebuild docker will be not save old version with the "None" name
* docker-compose --no-cache : rebuild not using cache

## mongodb
* Lets copy data to volume to a void reinitialize data from container

* Error when you start mongo service, lets set permission for log and data path 

```
cd /var/lib/mongodb
sudo chown -R mongodb:mongodb *
sudo chown -R mongodb:mongodb *
chown -Rc mongodb. /var/lib/mongodb
sudo chown -Rc mongodb. /var/lib/mongodb
sudo chown -Rc mongodb. /var/log/mongodb
```
* Initialize data in container 
basic method is volumes and restore data to container 

```
mongorestore folder_backup 
```
## dockerHub
first, you need to login docker
```
docker login 
```
create tag for images
```
docker tag image_name:image_tag your_ID /your-repository:tag
```
push image to dockerhub
```

docker push your_ID/your_repository:tag
```

****





![picture](/home/dsoft/Music/me/util_facelog/facelog_v2/1_02NqqST-JKZCDAPg9mOBiQ.png)