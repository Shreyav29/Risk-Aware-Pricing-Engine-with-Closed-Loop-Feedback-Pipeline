# Vertical Feedback Loop Diagram

```mermaid
flowchart TD
    A["Home + Macro Features"]
    B["Quantile AVM<br/>Q10, Q50, Q90"]
    C["Width Calibration<br/>Reliable Valuation Band"]
    D["Risk-Adjusted Buy Offer<br/>Base Spread + Uncertainty Penalty"]
    E["Home Resells"]
    F["Sell-Side Outcome<br/>Actual / Simulated Sale Price"]
    G["Market Multiplier<br/>Median Sale Ratio"]
    H["Feedback-Adjusted Quantiles<br/>Q10, Q50, Q90 shifted up/down"]
    I["Next Customer Offer"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    C -. "calibrated band" .-> H
    H --> I

    classDef data fill:#EEF2FF,stroke:#4F46E5,stroke-width:1.5px,color:#111827
    classDef model fill:#F0FDFA,stroke:#0F766E,stroke-width:1.5px,color:#111827
    classDef calibration fill:#FFF7ED,stroke:#C2410C,stroke-width:1.5px,color:#111827
    classDef offer fill:#F7FEE7,stroke:#4D7C0F,stroke-width:1.5px,color:#111827
    classDef sell fill:#FDF2F8,stroke:#BE185D,stroke-width:1.5px,color:#111827
    classDef feedback fill:#EFF6FF,stroke:#1D4ED8,stroke-width:2px,color:#111827

    class A data
    class B model
    class C calibration
    class D,I offer
    class E,F sell
    class G,H feedback
```

## Slide Takeaway

```text
The AVM estimates value and uncertainty.
The offer engine prices that uncertainty.
Sell-side outcomes shift the next valuation band through a market multiplier.
```
