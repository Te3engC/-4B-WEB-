# 在树莓派4B上实现人脸识别和部署WEB服务器

## 人脸识别

### 把树莓派apt源换成国内的源

清华源和阿里的源都可以 源自己找

```shell
sudo nano /etc/apt/sources.list.d/raspi.list
```

```shell
sudo nano /etc/apt/sources.list
```

### 安装opencv

- 更新apt-get

```shell
sudo apt-get update
sudo apt-get upgrade
```

- 安装opencv

```shell
sudo apt-get install python3-opencv
```

- 安装后验证

```shell
cv2.__version__
sudo apt install -y python3-opencv
```

### 测试CSI官方摄像头是否可以使用

- 注意摄像头必须插在CAMERA处

![IMG_5336(1)](D:\QQ存储\IMG_5336(1).jpg)

```shell
rpicam-hello
```

Raspberry Pi OS _Bookworm_ 将摄像头捕捉应用程序从 `libcamera-\*` 更名为 `rpicam-*`。符号链接允许用户暂时使用旧名称。尽快采用新的应用程序名称。 _Bookworm_之前的 Raspberry Pi OS 版本仍使用 `libcamera-*` 名称。我们下载的版本是新版已经不使用libcamera版本了

### 使用OPENCV实现人脸识别

```python
import cv2
import os
import pickle
import numpy as np
from picamera2 import Picamera2, Preview

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError("error")

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

feature_file = "face_features.pkl"

if os.path.exists(feature_file):
    with open(feature_file, "rb") as f:
        known_features, known_names = pickle.load(f)
else:
    known_features, known_names = [], []

def register_face(name, num_photos=5):
    path = f"known_faces/{name}"
    os.makedirs(path, exist_ok=True)
    count = 0
    while count < num_photos:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{path}/{count}.jpg", face_img)
            kp, des = orb.detectAndCompute(face_img, None)
            if des is not None:
                known_features.append(des)
                known_names.append(name)
                count += 1
                print(f"Saved face {count} for {name}")
        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    with open(feature_file, "wb") as f:
        pickle.dump((known_features, known_names), f)
    print(f"{name} registration complete and features saved.")
#设置你的面部信息
#register_face("tesengc", num_photos=10)

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            kp, des = orb.detectAndCompute(face_img, None)
            label = "Unknown"
            if des is not None:
                best_score = 0
                for i, known_des in enumerate(known_features):
                    matches = bf.match(des, known_des)
                    score = len(matches)
                    if score > best_score:
                        best_score = score
                        label = known_names[i]
                if best_score < 10:
                    label = "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()

```

### 使用方法

先把注册注释取消了，注册你的面部信息

![365e587c-2cc3-4ebf-ab1a-0fb52b35bacc](file:///C:/Users/94807/Pictures/Typedown/365e587c-2cc3-4ebf-ab1a-0fb52b35bacc.png)

![ace3112c-0eb5-4137-b7ba-a3be458537ba](file:///C:/Users/94807/Pictures/Typedown/ace3112c-0eb5-4137-b7ba-a3be458537ba.png)

重新注释开始识别

![2374745c-4927-4776-b481-e264895447b7](file:///C:/Users/94807/Pictures/Typedown/2374745c-4927-4776-b481-e264895447b7.png)

## 在树莓派上部署WEB服务器

### 设置静态IP

```shell
sudo nano /etc/dhcpcd.conf
```

```shell
    interface eth0
    static ip_address=192.168.137.233/24 
    static routers=192.168.137.1  
    static domain_name_servers=114.114.114.114 8.8.8.8

```

- 建议设置成连接VNC的IP 可以避免一些maf

### 重启

```shell
sudo reboot
```

### 设置时区

```shell
sudo dpkg-reconfigure tzdata
```

- 选择Asia

### 安装Ngnix/PHP/SQLite3

- 安装nginx

```shell
sudo apt-get install nginx
```

- 安装SQLite3

```shell
sudo apt-get install sqlite3
sudo apt install php8.4-sqlite3
sudo systemctl restart nginx
```

- 安装PHP8

```shell
sudo apt install -y php8.4-cli
sudo apt install -y php8.4-fpm php8.4-mysql php8.4-curl php8.4-gd php8.4-mbstring php8.4-xml php8.4-zip php8.4-json
```

- 验证

```shell
php -v
```

- 配置Nignx

```shell
sudo nano /etc/nginx/nginx.conf
```

```c
user www-data;
worker_processes 1; #修改这里
pid /var/run/nginx.pid;
events {
worker_connections 64; #修改这里
#multi_accept on;
}
```

- 修改gzip

```c
gzip on;
gzip_disable “msie6”;
gzip_vary on;
gzip_proxied any;
gzip_comp_level 6;
gzip_buffers 16 8k;
gzip_http_version 1.1;
gzip_types text/plain text/css application/json application/x-javascript text/xml application/xml application/xml+rss text/javascript;

```

### 配置PHP 注意版本根据你的版本改就好，这只是一个建议

```shell
sudo nano /etc/php/8.4/fpm/php.ini
```

```c
; The maximum number of processes FPM will fork. This has been design to control
; the global number of processes when using dynamic PM within a lot of pools.
; Use it with caution.
; Note: A value of 0 indicates no limit
; Default Value: 0
process.max = 4 #修改这里
```

```shell
sudo nano /etc/nginx/sites-available/default
```

- 测试

```shell
sudo nano /var/www/html/index.php
```

```php
<?php
      phpinfo();
?>


```

- 重启

```shell
sudo /etc/init.d/nginx restart
sudo /etc/init.d/php7.0-fpm restart

```

![6e4fbab51dac447c9b101af8b71835ee.png~tplv-a9rns2rl98-image](C:\Users\94807\Desktop\112200406陈增嵌入式实践\6e4fbab51dac447c9b101af8b71835ee.png~tplv-a9rns2rl98-image.png) 

### 安装typecho

如果可以直接下载就apt下载，不行的话可以先下载，传输给树莓派

[typecho/typecho: A PHP Blogging Platform. Simple and Powerful.](https://github.com/typecho/typecho)

```shell
cd /home/pi

sudo mv typecho-master.zip /var/www/html

cd /var/www/html
sudo unzip typecho-master.zip
sudo mv typecho-master/* .
sudo rm -rf typecho-master typecho-master.zip


sudo chown -R www-data:www-data /var/www/html
sudo chmod -R 755 /var/www/html
```

- 根据指示安装typecho

在主机上输入树莓派的静态IP

![3e39a5dc-5337-4750-b8a0-45c945a956f8](file:///C:/Users/94807/Pictures/Typedown/3e39a5dc-5337-4750-b8a0-45c945a956f8.png)

![1a2ff887-941d-45ed-90a2-05209480011f](file:///C:/Users/94807/Pictures/Typedown/1a2ff887-941d-45ed-90a2-05209480011f.png)

![e257bc81-4ee0-4fa1-9485-a848897ac8aa](file:///C:/Users/94807/Pictures/Typedown/e257bc81-4ee0-4fa1-9485-a848897ac8aa.png)

# 建议

善用搜索软件和AI

# 参考

[<u><span class="15">2024.10.1 多种方法在树莓派4B bookworm 上安装OPENCV_树莓派安装opencv-CSDN博客</span></u>](https://blog.csdn.net/2403_83160343/article/details/142674933)

[<u><span class="15">Raspbian Stretch：在你的Raspberry Pi上安装OpenCV 3 + Python_raspbian stretch: install opencv 3 + python on you-CSDN博客</span></u>](https://blog.csdn.net/LIEVE_Z/article/details/79899685)

[<u><span class="15"><font face="宋体">摄像头软件</font> | Raspberry Pi 树莓派 (官网25年12月更新)</span></u>](https://pidoc.cn/docs/computers/camera-software/)

[<u><span class="15"><font face="宋体">用树莓派实现实时的人脸检测</font> <font face="宋体">– 树莓派中文站</font></span></u>](http://www.52pi.net/archives/1435)

[<u><span class="15"><font face="宋体">树莓派</font>4B，5B换清华源 （最新版）【2025.2.28】_树莓派4b换源-CSDN博客</span></u>](https://blog.csdn.net/2302_81218871/article/details/145938245)

[<u><span class="15"><font face="宋体">利用树莓派搭建</font> web 服务器 (个人认为是网上步骤最全，也是最新的方式了 使用 PHP7)_树莓派web服务器-CSDN博客</span></u>](https://blog.csdn.net/qq_39125451/article/details/84898288)

[<u><span class="15">typecho/typecho: A PHP Blogging Platform. Simple and Powerful.</span></u>](https://github.com/typecho/typecho)


