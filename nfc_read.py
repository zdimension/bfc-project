import nfc
from nfc.clf import RemoteTarget

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
                print(tag)
            else:
                print("removed")
