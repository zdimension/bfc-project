# DoorPi

## Description
DoorPi is a raspberry image and python software developped to:

* Wait for an order from rasp-manager to turn on the camera
* compute a face detection model to take a picture of a person
* Send the results to rasp-manager for face recognition

##   Prerequisites

Use the image provided in the previous readme unless you wanna deal with long hours of configuration problems ;)

In order to run our application on any raspberry pi, you will need to:

* Have a Rasbian OS installed on your raspberry pi (32 bits Bullseye)
* Have a supplicant.conf file with the wifi credentials of Rasp-Manager
* Have a camera module installed on your raspberry pi
* Have a NFC reader installed on your raspberry pi
* Have configured first the rasp-manager server with the running access point.
* Have python version 3.7.12 installed, you can use pyenv to make the installation easily. (It doesn't work with python 3.9 because tensorflow is not compatible with it)
* transfer the DoorPi folder to your raspberry pi
* Create a virtual environment with python 3.7.12
* install the requirements.txt file with pip
* Move the rasp-manager certificate to /usr/share/ca-certificates/cert.pem
* You may need to update the hosts file of DoorPi with the ip_adress of rasp-manager and its domain name like in the certificate. (because it is self signed)

## Run

If you want to run the DoorPi server (without the service/testing purposes) :


```uvicorn DoorPi:app --host 0.0.0.0 --port 8000 --ssl-keyfile ./key.pem --ssl-certfile ./cert.pem```

You will need to enter the password of your cert file.
the provided one for testing purposes is ```bfc-a-team```.

If you want to run the DoorPi server with the on-boot service :

* Move the .service files to /etc/systemd/system and enable them with systemctl (this will allow the program to run on boot)

* Start the service with systemctl

* You can check the logs with :   
    ```journalctl -u DoorPi.service```  
    ```journalctl -u nfc.service```


