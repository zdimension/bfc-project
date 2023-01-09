# bfc-project

## Architecture

```mermaid
graph LR
    F[LAN] --> |Ethernet| A
    A["RPi\n(server)"] --> |Wi-Fi| B["RPi\n(door)"]
    B --> |USB| C[Webcam]
    B --> |USB| D[NFC reader]
    B --> |tbd| E[Door lock]
    B --> |GPIO| G[Status LEDs]
    A --> |Wi-Fi| H[User device]

    subgraph " "
        C;D;E;G;B
    end
```
