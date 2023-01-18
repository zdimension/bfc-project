import sys
def main():
    #verify that there is enough arguments
    if len(sys.argv) < 2:
        print("Not enough arguments")
        return
    ip = sys.argv[1]
    #overwrite the hosts file with the new one
    hosts_path = "/etc/hosts"
    data = "127.0.0.1	localhost\n"\
    "::1		localhost ip6-localhost ip6-loopback\n"\
    "ff02::1		ip6-allnodes\n"\
    "ff02::2		ip6-allrouters\n"\
    "127.0.1.1		raspberrypi\n"\
    ""+ ip + "		DoorPi" 
    with open(hosts_path, 'w') as file:
        file.write(data)

if __name__ == "__main__":
    main()
