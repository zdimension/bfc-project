#!/usr/bin/env python3.7
import sys
sys.path.append("/home/pi/DoorPi/env/lib/python3.7/site-packages/")
import nfc
from nfc.clf import RemoteTarget
import requests


with nfc.ContactlessFrontend("usb") as clf:
    target = None
    while True:
        new_target = clf.sense(
            RemoteTarget("106A"),
            RemoteTarget("106B"),
            RemoteTarget("212F")
        )
        if type(new_target) != type(target):
            target = new_target
            if target:
                tag = nfc.tag.activate(clf, target)
                r = requests.post('https://rasp-manager:8000/NfcVerification', json={'personTag': repr(tag)}, verify="/usr/share/ca-certificates/cert.pem")
                print(repr(tag))
                print(tag.dump())
                if tag.ndef is not None:
                    print(repr(tag.ndef.records))
                print(r.text)
            else:
                print("removed")
