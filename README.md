# bfc-project

## Architecture

```mermaid
graph LR
    F[LAN] --- |Ethernet| A
    A["RPi<br>(server)"] --- |Wi-Fi| B["RPi<br>(door)"]
    B --- |USB| C[Webcam]
    B --- |USB| D[NFC reader]
    B --- |tbd| E[Door lock]
    B --- |GPIO| G[Status LEDs]
    A --- |Wi-Fi| H[User device]

    subgraph " "
        C;D;E;G;B
    end
```
