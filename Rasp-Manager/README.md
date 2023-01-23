# DoorPi

## Description
Rasp-Manager is a raspberry image and python software developped to:

* Initiate a wifi access point
* Use NFC UID to verify a person identity
* Use face recognition to compare identity with the person in front of the camera
* Can open a door 
* Provide internet to the DoorPi if connected to the router of your house by ethernet

##   Prerequisites

Use the image provided in the previous readme unless you wanna deal with long hours of configuration problems ;) (do not suffer like I did for this) (it was fun though)

In order to run our application on any raspberry pi, you will need to:

* Have a Rasbian OS installed on your raspberry pi (64 bits Bullseye)
* transfer the Rasp-Manager folder to your raspberry pi
* Create a virtual environment with python
* install the requirements.txt file with pip
* Move the DoorPi certificate to /usr/share/ca-certificates/cert.pem
* Do all the requirements to have the access point working on boot(see the following tutorial https://www.youtube.com/watch?v=S4E35d91Xss)

## Run

If you want to run the rasp-manager server (without the service/testing purposes) :


```uvicorn piserver:app --host 0.0.0.0 --port 8000 --ssl-keyfile ./key.pem --ssl-certfile ./cert.pem```

You will need to enter the password of your cert file.
the provided one for testing purposes is ```bfc-a-team```

If you want to run the rasp-manager server with the on-boot service :

* Move the .service files to /etc/systemd/system and enable them with systemctl (this will allow the program to run on boot)

* Start the service with systemctl

* You can check the logs with 
    ```journalctl -u rasp-manager.service```

## Misc

You may need to use the /init route with the DoorPi ip_adress. When you contact the route, a small programm "updateHosts" is ran to update the hosts file of your computer with the ip_adresses of the DoorPi.
We had to do this because it's a self signed certificate and the browser doesn't like it.
