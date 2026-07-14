# Road Incident Detection System — UML Activity Diagram

Black-and-white UML activity diagram, portrait 3:4, laid out to fill the page.

![System UML](system_uml.png)

> For infinite sharpness open `system_uml.svg` (vector, never blurs at any zoom).

## How to read it

The frame is processed **once** (detect + track), then a **fork** fans it out to four detection
heads that run **in parallel**. A **join** waits for all of them, candidate events are fused and
correlated, and only if a flag clears the threshold does a **human** confirm or dismiss. The
system never acts on its own.

### The accident head — how it avoids false alarms in traffic

- **Contact + shock** — two boxes overlap (IoU) and a moving vehicle suddenly loses most of its speed.
- **Static** — after impact the boxes go still. The catch: **a traffic jam also freezes the boxes.**
- **Disambiguate** — *crash* (odd angle, boxes merged, car stays at rest) vs *traffic*
  (lane-aligned, gradual → ignored).
- **Flag** — backdate to the **first frame of contact** and record the involved IDs.

## Build status

| Part | Status | Tech |
|---|---|---|
| Detect + track (shared backbone) | **Built** | YOLO11x (pretrained) + ByteTrack |
| Accident / collision | **Built** | IoU + shock + static-vs-traffic + rest-confirm, two-pass |
| Fight / road-rage | Planned | body pose over time + temporal action model (RWF-2000) |
| Weapon / gun | Planned | weapon detector fine-tuned + persist-across-frames |
| Crowd / mob | Planned | person density map (CSRNet) + surge |
| Fusion + human review | Partial / Planned | correlate escalation chain, operator confirms, never auto-acts |
| LLM / VLM verification | Planned (optional) | vision-language model confirms real events; slots between the threshold and the alert |

**The honest gate:** accidents work today (pretrained detection + tuned rules). The other heads are
*learned* models — they need labelled domain data to be trustworthy.

## Two design rules

- **One detect+track pass feeds every head.** Do not run four models per frame.
- **Human-in-the-loop, always.** Every output is an assistive flag with a confidence, never an action.

<details>
<summary>Diagram source (renders on GitHub / Notion / VS Code)</summary>

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontFamily':'Segoe UI','primaryColor':'#ffffff','primaryTextColor':'#111111','primaryBorderColor':'#111111','lineColor':'#111111','tertiaryColor':'#ffffff','fontSize':'18px'}, 'flowchart': {'nodeSpacing': 172, 'rankSpacing': 20, 'padding':6, 'curve': 'linear'}}}%%
flowchart TD
    START((Start)) --> ING("Ingest frame<br/>(CCTV / dashcam)")
    ING --> DET("Detect people + vehicles<br/>(YOLO11x)")
    DET --> TRK("Track stable IDs<br/>(ByteTrack)")
    TRK --> FORK["Fork &mdash; run heads in parallel"]

    FORK --> ACC1("ACCIDENT (built)<br/>Per-vehicle speed from motion")
    FORK --> FGT1("FIGHT / ROAD-RAGE (planned)<br/>Track body pose over time")
    FORK --> GUN1("WEAPON / GUN (planned)<br/>Weapon detector on crops")
    FORK --> CRD1("CROWD / MOB (planned)<br/>Person density map (CSRNet)")

    ACC1 --> ACC2{"Boxes overlap (IoU) +<br/>sudden speed loss?"}
    ACC2 -->|No| JOIN["Join &mdash; wait for all"]
    ACC2 -->|Yes| ACC3("Vehicles go static<br/>box coords freeze")
    ACC3 --> ACC4{"Static = crash<br/>or traffic?"}
    ACC4 -->|Traffic| JOIN
    ACC4 -->|Crash| ACC5("Crash: odd angle,<br/>boxes merged")
    ACC5 --> ACC6{"Car at rest at<br/>impact >= N frames?"}
    ACC6 -->|No| JOIN
    ACC6 -->|Yes| AF("Flag ACCIDENT<br/>backdate to first contact + IDs")
    AF --> JOIN

    FGT1 --> FGT2{"Action model flags<br/>aggression? (RWF-2000)"}
    FGT2 -->|Yes| FF("Flag FIGHT")
    FGT2 -->|No| JOIN
    FF --> JOIN

    GUN1 --> GUN2{"Gun persists across<br/>several frames?"}
    GUN2 -->|Yes| GF("Flag WEAPON")
    GUN2 -->|No| JOIN
    GF --> JOIN

    CRD1 --> CRD2{"Sudden surge or<br/>abnormal gathering?"}
    CRD2 -->|Yes| CF("Flag CROWD")
    CRD2 -->|No| JOIN
    CF --> JOIN

    JOIN --> FUSE("Fuse + correlate events<br/>escalation: Crash &rarr; Fight &rarr; Gun")
    FUSE --> Q{"Any flag over<br/>threshold?"}
    Q -->|Yes| AL("Alert operator<br/>clip + confidence")
    AL --> HUM("Human confirms / dismisses<br/>system never auto-acts")
    HUM --> OUT("Save annotated video<br/>+ events.json")
    Q -->|No| MON("Keep monitoring")
    OUT --> END((End))
    MON --> END

    classDef term fill:#111111,stroke:#111111,color:#ffffff;
    classDef bar fill:#111111,stroke:#111111,color:#ffffff;
    classDef act fill:#ffffff,stroke:#111111,color:#111111,stroke-width:1.4px;
    classDef dec fill:#ffffff,stroke:#111111,color:#111111,stroke-width:1.4px;

    class START,END term;
    class FORK,JOIN bar;
    class ING,DET,TRK,ACC1,ACC3,ACC5,AF,FGT1,GUN1,CRD1,FF,GF,CF,FUSE,AL,HUM,OUT,MON act;
    class ACC2,ACC4,ACC6,FGT2,GUN2,CRD2,Q dec;
```
</details>
