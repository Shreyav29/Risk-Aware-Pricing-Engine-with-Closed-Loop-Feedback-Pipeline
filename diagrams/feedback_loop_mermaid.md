# Sell-Side Feedback Loop Diagram

```mermaid
flowchart TD
    subgraph DATA["Data + Market Context"]
        A["King County Sales<br/>Property Attributes"]
        B["FRED Macro Features<br/>Rates, unemployment, supply"]
    end

    subgraph MODEL["Probabilistic AVM"]
        C["Feature Engineering"]
        D["Quantile AVM"]
        E["Raw Quantiles<br/>Q10_raw, Q50_raw, Q90_raw"]
    end

    subgraph CAL["Calibration Layer"]
        F["Width Calibration<br/>learned on validation set"]
        G["Calibrated Band<br/>Q10_cal, Q50, Q90_cal"]
    end

    subgraph OFFER1["Initial Buy-Side Offer"]
        H["Uncertainty Score<br/>(Q90_cal - Q10_cal) / Q50"]
        I["Risk-Adjusted Offer<br/>base spread + uncertainty penalty"]
    end

    subgraph SELL["Sell-Side Observation"]
        J["Expected Resale Value<br/>Q50"]
        K["Observed / Simulated Resale<br/>actual price * stress factor"]
        L["Sale Ratio<br/>observed resale / expected Q50"]
    end

    subgraph FEEDBACK["Feedback Adjustment"]
        M["Market Multiplier<br/>median sale ratio"]
        N["Shift Calibrated Band<br/>Q10_fb = Q10_cal * m<br/>Q50_fb = Q50 * m<br/>Q90_fb = Q90_cal * m"]
    end

    subgraph OFFER2["Next Buy-Side Offer"]
        O["Updated Offer Engine<br/>same spread logic"]
        P["Next Customer Offer<br/>adjusted for market bias"]
    end

    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I

    G --> J
    K --> L
    J --> L
    L --> M
    G --> N
    M --> N
    N --> O
    O --> P

    classDef data fill:#EEF2FF,stroke:#4F46E5,stroke-width:1.5px,color:#111827
    classDef model fill:#F0FDFA,stroke:#0F766E,stroke-width:1.5px,color:#111827
    classDef calibration fill:#FFF7ED,stroke:#C2410C,stroke-width:1.5px,color:#111827
    classDef offer fill:#F7FEE7,stroke:#4D7C0F,stroke-width:1.5px,color:#111827
    classDef sell fill:#FDF2F8,stroke:#BE185D,stroke-width:1.5px,color:#111827
    classDef feedback fill:#EFF6FF,stroke:#1D4ED8,stroke-width:2px,color:#111827

    class A,B data
    class C,D,E model
    class F,G calibration
    class H,I,O,P offer
    class J,K,L sell
    class M,N feedback
```

## Key Interpretation

The feedback loop does **not** retrain the AVM and does **not** change raw model
parameters. It applies a post-calibration market-level correction:

```text
Raw quantiles
-> width-calibrated quantiles
-> sell-side market multiplier
-> feedback-adjusted quantiles
-> next buy offer
```

This keeps the two corrections separate:

```text
Width calibration = property-level uncertainty reliability
Market multiplier = market-level bias correction
```
