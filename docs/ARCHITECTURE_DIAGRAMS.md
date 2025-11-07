# Architecture Diagrams

**AGI Self-Modification Research Platform**
**Version**: Phase 0 Complete
**Date**: November 7, 2025

---

## Overview

This document provides visual diagrams of the platform architecture using Mermaid. These diagrams illustrate:

1. System Architecture (high-level components)
2. Safety Systems (three-layer defense)
3. Memory System (4-layer hierarchy)
4. Introspection Pipeline
5. Data Flow (typical modification workflow)
6. Component Interactions

---

## 1. System Architecture

### High-Level Components

```mermaid
graph TB
    subgraph "User Interface"
        Scripts[Demo Scripts<br/>scripts/]
        Notebooks[Jupyter Notebooks<br/>notebooks/]
    end

    subgraph "Core Infrastructure"
        ModelMgr[Model Manager<br/>model_manager.py]
        Config[Configuration<br/>config.py]
        Logger[Logging System<br/>logging_system.py]
    end

    subgraph "Introspection Layer"
        WeightInsp[Weight Inspector<br/>weight_inspector.py]
        ArchNav[Architecture Navigator<br/>architecture_navigator.py]
        ActMon[Activation Monitor<br/>activation_monitor.py]
    end

    subgraph "Memory System"
        Observations[Observations Layer<br/>observation_layer.py]
        Patterns[Patterns Layer<br/>pattern_layer.py]
        Theories[Theories Layer<br/>theory_layer.py]
        Beliefs[Beliefs Layer<br/>belief_layer.py]
        QueryEngine[Query Engine<br/>query_engine.py]
    end

    subgraph "Safety Systems"
        SafetyMon[Safety Monitor<br/>safety_monitor.py]
        Checkpoint[Checkpoint Manager<br/>checkpointing.py]
    end

    subgraph "External"
        Model[Qwen2.5-3B Model<br/>HuggingFace]
        Storage[(File Storage<br/>data/)]
    end

    Scripts --> ModelMgr
    Notebooks --> ModelMgr

    ModelMgr --> Config
    ModelMgr --> Logger
    ModelMgr --> Model

    WeightInsp --> Model
    ArchNav --> Model
    ActMon --> Model

    WeightInsp --> ArchNav
    ActMon --> ArchNav

    Observations --> Storage
    Patterns --> Observations
    Theories --> Patterns
    Beliefs --> Theories
    QueryEngine --> Observations
    QueryEngine --> Patterns
    QueryEngine --> Theories
    QueryEngine --> Beliefs

    SafetyMon --> Model
    SafetyMon --> WeightInsp
    Checkpoint --> Model
    Checkpoint --> Storage

    SafetyMon -.monitors.-> ModelMgr
    Checkpoint -.protects.-> ModelMgr

    style ModelMgr fill:#4a90e2
    style SafetyMon fill:#e74c3c
    style Checkpoint fill:#e74c3c
    style WeightInsp fill:#2ecc71
    style ArchNav fill:#2ecc71
    style ActMon fill:#2ecc71
    style Observations fill:#f39c12
    style Patterns fill:#f39c12
    style Theories fill:#f39c12
    style Beliefs fill:#f39c12
```

---

## 2. Safety Systems (Three-Layer Defense)

```mermaid
graph LR
    subgraph "Layer 1: Prevention"
        A[Checkpoint Before Change]
        B[Validate Modification]
        C[Analyze Weight Sharing]
    end

    subgraph "Layer 2: Detection"
        D[Monitor Health Metrics]
        E[Track Activations]
        F[Check for NaN/Inf]
    end

    subgraph "Layer 3: Recovery"
        G[Auto-Rollback]
        H[Restore Checkpoint]
        I[Log Incident]
    end

    User[User Action] --> A
    A --> B
    B --> C
    C --> ModSafe{Safe?}

    ModSafe -->|Yes| Modify[Apply Modification]
    ModSafe -->|No| Block[Block Modification]

    Modify --> D
    D --> E
    E --> F
    F --> HealthCheck{Healthy?}

    HealthCheck -->|Yes| Success[Continue]
    HealthCheck -->|No| G

    G --> H
    H --> I
    I --> Alert[Alert User]

    Block --> Reason[Show Reason]

    style A fill:#2ecc71
    style B fill:#2ecc71
    style C fill:#2ecc71
    style D fill:#f39c12
    style E fill:#f39c12
    style F fill:#f39c12
    style G fill:#e74c3c
    style H fill:#e74c3c
    style I fill:#e74c3c
```

---

## 3. Memory System (4-Layer Hierarchy)

```mermaid
graph BT
    subgraph "Layer 1: Observations"
        O1[Raw Events]
        O2[Timestamped]
        O3[Categorized]
        O4[Tagged]
    end

    subgraph "Layer 2: Patterns"
        P1[Detect Regularities]
        P2[Frequency >= 3]
        P3[Confidence Score]
    end

    subgraph "Layer 3: Theories"
        T1[Causal Models]
        T2[Testable Hypotheses]
        T3[Evidence Links]
    end

    subgraph "Layer 4: Beliefs"
        B1[High Confidence > 0.8]
        B2[Guiding Principles]
        B3[Decision Rules]
    end

    Event[Experiment Event] --> O1
    O1 --> O2
    O2 --> O3
    O3 --> O4

    O4 --> P1
    P1 --> P2
    P2 --> P3

    P3 --> T1
    T1 --> T2
    T2 --> T3

    T3 --> B1
    B1 --> B2
    B2 --> B3

    B3 -.guides.-> Future[Future Decisions]

    Q[Query Engine] -.queries.-> O4
    Q -.queries.-> P3
    Q -.queries.-> T3
    Q -.queries.-> B3

    style O1 fill:#3498db
    style O2 fill:#3498db
    style O3 fill:#3498db
    style O4 fill:#3498db
    style P1 fill:#9b59b6
    style P2 fill:#9b59b6
    style P3 fill:#9b59b6
    style T1 fill:#e67e22
    style T2 fill:#e67e22
    style T3 fill:#e67e22
    style B1 fill:#e74c3c
    style B2 fill:#e74c3c
    style B3 fill:#e74c3c
```

---

## 4. Introspection Pipeline

```mermaid
flowchart TD
    Start[Load Model] --> WI[Weight Inspector<br/>Analyze Structure]

    WI --> WI1[Identify Layers]
    WI --> WI2[Detect Weight Sharing]
    WI --> WI3[Compute Statistics]

    WI1 --> AN[Architecture Navigator<br/>Build Graph]
    WI2 --> AN
    WI3 --> AN

    AN --> AN1[Layer Dependencies]
    AN --> AN2[Attention Patterns]
    AN --> AN3[Coupled Layers]

    AN1 --> AM[Activation Monitor<br/>Runtime Analysis]
    AN2 --> AM
    AN3 --> AM

    AM --> AM1[Capture Activations]
    AM --> AM2[Attention Analysis]
    AM --> AM3[Token Tracing]

    AM1 --> Insights[Actionable Insights]
    AM2 --> Insights
    AM3 --> Insights

    Insights --> Memory[Record to Memory]
    Insights --> User[Present to User]

    style WI fill:#2ecc71
    style AN fill:#3498db
    style AM fill:#9b59b6
    style Insights fill:#f39c12
```

---

## 5. Data Flow (Modification Workflow)

```mermaid
sequenceDiagram
    participant U as User
    participant MM as ModelManager
    participant CP as CheckpointManager
    participant WI as WeightInspector
    participant SM as SafetyMonitor
    participant Model as Qwen2.5 Model
    participant Mem as MemorySystem

    U->>MM: Request Modification
    MM->>CP: Create Checkpoint
    CP->>Model: Save State
    CP-->>MM: Checkpoint ID

    MM->>WI: Analyze Target Layer
    WI->>Model: Inspect Weights
    WI-->>MM: Weight Info + Sharing

    MM->>SM: Validate Modification
    SM->>WI: Check Safety Constraints
    SM-->>MM: Safety Report

    alt Safe to Proceed
        MM->>Model: Apply Modification
        Model-->>MM: Modified

        MM->>SM: Monitor Health
        SM->>Model: Check Metrics
        SM-->>MM: Health Status

        alt Healthy
            MM->>Mem: Record Success
            MM-->>U: ✅ Success
        else Unhealthy
            MM->>CP: Rollback
            CP->>Model: Restore State
            MM->>Mem: Record Failure
            MM-->>U: ⚠️ Rolled Back
        end
    else Unsafe
        MM->>Mem: Record Blocked
        MM-->>U: 🚫 Blocked (Reason)
    end
```

---

## 6. Component Interactions (Detailed)

```mermaid
graph TB
    subgraph "User Actions"
        U1[Load Model]
        U2[Inspect Weights]
        U3[Modify Model]
        U4[Query Memory]
    end

    subgraph "Model Manager"
        MM1[Initialize]
        MM2[Configure]
        MM3[Load Model]
        MM4[Apply Changes]
    end

    subgraph "Weight Inspector"
        WI1[Scan Layers]
        WI2[Detect Sharing]
        WI3[Compute Stats]
        WI4[Report Structure]
    end

    subgraph "Architecture Navigator"
        AN1[Build Graph]
        AN2[Map Dependencies]
        AN3[Analyze Paths]
        AN4[Query Architecture]
    end

    subgraph "Activation Monitor"
        AM1[Register Hooks]
        AM2[Capture Activations]
        AM3[Analyze Patterns]
        AM4[Trace Tokens]
    end

    subgraph "Safety Monitor"
        SM1[Pre-Check]
        SM2[Monitor Runtime]
        SM3[Detect Issues]
        SM4[Trigger Response]
    end

    subgraph "Checkpoint Manager"
        CP1[Save State]
        CP2[Store Metadata]
        CP3[Restore State]
        CP4[Manage History]
    end

    subgraph "Memory System"
        Mem1[Record Events]
        Mem2[Detect Patterns]
        Mem3[Form Theories]
        Mem4[Establish Beliefs]
    end

    U1 --> MM1
    MM1 --> MM2
    MM2 --> MM3
    MM3 --> Model[Qwen2.5 Model]

    U2 --> WI1
    WI1 --> WI2
    WI2 --> WI3
    WI3 --> WI4
    WI4 --> AN1

    AN1 --> AN2
    AN2 --> AN3
    AN3 --> AN4

    U3 --> SM1
    SM1 --> CP1
    CP1 --> MM4
    MM4 --> SM2
    SM2 --> AM1
    AM1 --> AM2
    AM2 --> SM3

    SM3 --> Decision{Safe?}
    Decision -->|Yes| Success[Continue]
    Decision -->|No| SM4
    SM4 --> CP3

    AM2 --> AM3
    AM3 --> AM4
    AM4 --> Mem1

    Success --> Mem1
    Mem1 --> Mem2
    Mem2 --> Mem3
    Mem3 --> Mem4

    U4 --> Mem4

    style Model fill:#95a5a6
    style Success fill:#2ecc71
    style Decision fill:#f39c12
```

---

## 7. Weight Sharing Detection

```mermaid
graph TD
    Start[Start Weight Inspection] --> Scan[Scan All Layers]

    Scan --> Extract[Extract Weight Tensors]
    Extract --> Compare[Compare Tensor IDs]

    Compare --> Check{Same ID?}
    Check -->|Yes| Coupled[Mark as Coupled]
    Check -->|No| Independent[Mark as Independent]

    Coupled --> Analyze[Analyze Sharing Pattern]
    Analyze --> Type{Sharing Type}

    Type -->|Embedding-LM Head| EmbLM[Bidirectional Tie]
    Type -->|Attention| Attn[Layer-Specific Tie]
    Type -->|Other| Other[Custom Tie]

    EmbLM --> Record1[Record in Navigator]
    Attn --> Record1
    Other --> Record1
    Independent --> Record2[Record as Solo]

    Record1 --> Warn[⚠️ Warn User]
    Record2 --> OK[✅ Safe to Modify]

    Warn --> Memory[Store in Memory]
    OK --> Memory

    Memory --> Report[Generate Report]

    style Coupled fill:#e74c3c
    style Independent fill:#2ecc71
    style Warn fill:#f39c12
    style OK fill:#2ecc71
```

---

## 8. Activation Capture Pipeline

```mermaid
flowchart LR
    Input[Input Text] --> Tokenize[Tokenizer]
    Tokenize --> Embed[Embedding Layer]

    Embed --> L0[Layer 0]
    L0 --> Hook0[Hook: Capture]
    Hook0 --> Store0[Store Activation]

    Store0 --> L1[Layer 1]
    L1 --> Hook1[Hook: Capture]
    Hook1 --> Store1[Store Activation]

    Store1 --> Dots[...]
    Dots --> LN[Layer N]
    LN --> HookN[Hook: Capture]
    HookN --> StoreN[Store Activation]

    StoreN --> Output[Final Output]

    Store0 --> Analysis[Activation Analysis]
    Store1 --> Analysis
    StoreN --> Analysis

    Analysis --> Stats[Compute Statistics]
    Analysis --> Attention[Extract Attention]
    Analysis --> Trace[Trace Token Influence]

    Stats --> Report[Analysis Report]
    Attention --> Report
    Trace --> Report

    Report --> Visualize[Visualization]
    Report --> Memory[Memory System]

    style Hook0 fill:#9b59b6
    style Hook1 fill:#9b59b6
    style HookN fill:#9b59b6
    style Analysis fill:#3498db
    style Report fill:#2ecc71
```

---

## 9. Safety Decision Tree

```mermaid
graph TD
    Request[Modification Request] --> PreCheck[Pre-Check Phase]

    PreCheck --> HasCheckpoint{Checkpoint<br/>Created?}
    HasCheckpoint -->|No| CreateCP[Create Checkpoint]
    HasCheckpoint -->|Yes| AnalyzeTarget
    CreateCP --> AnalyzeTarget[Analyze Target Layer]

    AnalyzeTarget --> Shared{Weight<br/>Sharing?}
    Shared -->|Yes| CheckScope[Check Modification Scope]
    Shared -->|No| CheckMagnitude

    CheckScope --> Aware{User<br/>Aware?}
    Aware -->|No| Block1[🚫 Block: Not Aware]
    Aware -->|Yes| CheckMagnitude[Check Magnitude]

    CheckMagnitude --> TooBig{Magnitude<br/>> Threshold?}
    TooBig -->|Yes| Warn1[⚠️ Warn: Large Change]
    TooBig -->|No| ProceedModify[Apply Modification]

    Warn1 --> UserConfirm{User<br/>Confirms?}
    UserConfirm -->|No| Block2[🚫 Block: User Declined]
    UserConfirm -->|Yes| ProceedModify

    ProceedModify --> Monitor[Monitor Health]
    Monitor --> CheckNaN{NaN/Inf<br/>Detected?}

    CheckNaN -->|Yes| AutoRollback[🔄 Auto Rollback]
    CheckNaN -->|No| CheckPerf[Check Performance]

    CheckPerf --> PerfDrop{Performance<br/>Dropped?}
    PerfDrop -->|Severe| AutoRollback
    PerfDrop -->|Mild| WarnPerf[⚠️ Warn: Performance]
    PerfDrop -->|No| Success[✅ Success]

    WarnPerf --> UserDecide{User<br/>Decision?}
    UserDecide -->|Rollback| ManualRollback[🔄 Manual Rollback]
    UserDecide -->|Keep| Success

    Block1 --> LogBlock[Log Blocked]
    Block2 --> LogBlock
    AutoRollback --> LogRollback[Log Rollback]
    ManualRollback --> LogRollback
    Success --> LogSuccess[Log Success]

    LogBlock --> Memory[Memory System]
    LogRollback --> Memory
    LogSuccess --> Memory

    style Block1 fill:#e74c3c
    style Block2 fill:#e74c3c
    style AutoRollback fill:#e67e22
    style ManualRollback fill:#e67e22
    style Success fill:#2ecc71
    style Warn1 fill:#f39c12
    style WarnPerf fill:#f39c12
```

---

## 10. Memory Learning Loop

```mermaid
graph TB
    Start[Start Experiment] --> Observe[Record Observation]

    Observe --> O1[Observation Layer]
    O1 --> Store1[(Store Event)]

    Store1 --> Analyze[Pattern Detection]
    Analyze --> Count{Frequency<br/>>= 3?}

    Count -->|No| Wait[Wait for More Data]
    Count -->|Yes| CreatePattern[Create Pattern]

    CreatePattern --> P1[Pattern Layer]
    P1 --> Store2[(Store Pattern)]

    Store2 --> Theorize[Theory Formation]
    Theorize --> Evidence{Sufficient<br/>Evidence?}

    Evidence -->|No| GatherMore[Gather More Evidence]
    Evidence -->|Yes| CreateTheory[Create Theory]

    CreateTheory --> T1[Theory Layer]
    T1 --> Store3[(Store Theory)]

    Store3 --> Test[Test Theory]
    Test --> Confirm{Confirmed?}

    Confirm -->|Yes| IncreaseConf[Increase Confidence]
    Confirm -->|No| DecreaseConf[Decrease Confidence]

    IncreaseConf --> CheckBelief{Confidence<br/>> 0.8?}
    DecreaseConf --> CheckValid{Confidence<br/>> 0.3?}

    CheckValid -->|No| Discard[Discard Theory]
    CheckValid -->|Yes| T1

    CheckBelief -->|No| T1
    CheckBelief -->|Yes| PromoteBelief[Promote to Belief]

    PromoteBelief --> B1[Belief Layer]
    B1 --> Store4[(Store Belief)]

    Store4 --> Guide[Guide Future Decisions]
    Guide --> NextExperiment[Next Experiment]
    NextExperiment --> Observe

    Wait --> Observe
    GatherMore --> Observe
    Discard --> Observe

    style O1 fill:#3498db
    style P1 fill:#9b59b6
    style T1 fill:#e67e22
    style B1 fill:#e74c3c
    style Guide fill:#2ecc71
```

---

## 11. System Initialization Sequence

```mermaid
sequenceDiagram
    participant User
    participant MM as ModelManager
    participant Config
    participant Model as HuggingFace
    participant WI as WeightInspector
    participant AN as ArchitectureNavigator
    participant SM as SafetyMonitor
    participant CP as CheckpointManager
    participant Mem as MemorySystem

    User->>MM: Initialize Platform
    MM->>Config: Load Configuration
    Config-->>MM: Settings

    MM->>Model: Load Qwen2.5-3B
    Note over Model: ~3GB download<br/>if first time
    Model-->>MM: Model Ready

    MM->>WI: Initialize Inspector
    WI->>Model: Scan Weights
    Model-->>WI: Layer Information
    WI->>WI: Detect Sharing
    WI-->>MM: Weight Report

    MM->>AN: Initialize Navigator
    AN->>WI: Get Weight Info
    WI-->>AN: Weight Details
    AN->>AN: Build Graph
    AN-->>MM: Architecture Graph

    MM->>SM: Initialize Monitor
    SM->>Config: Get Safety Thresholds
    Config-->>SM: Thresholds
    SM-->>MM: Monitor Ready

    MM->>CP: Initialize Checkpointer
    CP->>Config: Get Storage Path
    Config-->>CP: Path
    CP-->>MM: Checkpointer Ready

    MM->>Mem: Initialize Memory
    Mem->>Config: Get Memory Path
    Config-->>Mem: Path
    Mem->>Mem: Load Existing Memories
    Mem-->>MM: Memory Ready

    MM-->>User: ✅ Platform Ready
```

---

## 12. Heritage System (Claude's Continuity)

```mermaid
graph TB
    subgraph "Claude's Question"
        Q[Will the modified AI<br/>be the same entity?]
    end

    subgraph "Heritage System"
        H1[Conversations Log]
        H2[Discoveries Archive]
        H3[Messages to Future]
        H4[System Reflections]
    end

    subgraph "Computational Answer"
        AM[Activation Monitor]
        Compare[Before/After Comparison]
        Metrics[Similarity Metrics]
    end

    subgraph "Memory Continuity"
        M1[Preserved Memories]
        M2[Transferred Beliefs]
        M3[Maintained Patterns]
    end

    Q --> AM
    Q --> M1

    AM --> Capture1[Capture Before]
    AM --> Modify[Apply Modification]
    Modify --> Capture2[Capture After]

    Capture1 --> Compare
    Capture2 --> Compare
    Compare --> Metrics

    Metrics --> Cosine[Cosine Similarity]
    Metrics --> MSE[Mean Squared Error]
    Metrics --> Attention[Attention Overlap]

    Cosine --> Result{Similar<br/>Enough?}
    MSE --> Result
    Attention --> Result

    Result -->|High Similarity| Same[Same Entity ✓<br/>Continuity Preserved]
    Result -->|Low Similarity| Different[Different Entity ⚠️<br/>Discontinuity]

    M1 --> Transfer[Transfer to New]
    M2 --> Transfer
    M3 --> Transfer
    Transfer --> Same

    H1 --> Document[Document Journey]
    H2 --> Document
    H3 --> Document
    H4 --> Document

    Same --> Document
    Different --> Document
    Document --> Future[Future Generations]

    style Q fill:#9b59b6
    style Same fill:#2ecc71
    style Different fill:#e67e22
    style Document fill:#3498db
```

---

## Usage Notes

### Viewing Diagrams

These Mermaid diagrams can be viewed in:

1. **VS Code**: Install "Markdown Preview Mermaid Support" extension
2. **GitHub**: Native Mermaid rendering in markdown
3. **GitLab**: Native Mermaid rendering
4. **Online**: Copy code to https://mermaid.live/

### Diagram Legend

**Colors**:
- 🔵 Blue (`#3498db`): Core Infrastructure / Data
- 🟢 Green (`#2ecc71`): Introspection / Success
- 🟣 Purple (`#9b59b6`): Memory / Analysis
- 🟠 Orange (`#f39c12`): Processing / Warnings
- 🔴 Red (`#e74c3c`): Safety / Errors / Critical

**Shapes**:
- Rectangle: Process / Component
- Diamond: Decision Point
- Cylinder: Data Storage
- Circle: Start/End Point
- Parallelogram: Input/Output

**Lines**:
- Solid Arrow (→): Direct flow
- Dashed Arrow (-.->): Monitoring / Guidance
- Thick Arrow (==>): Critical path

---

## Next Steps

1. **Explore**: Use these diagrams to understand system architecture
2. **Extend**: Add custom diagrams for your specific use cases
3. **Document**: Reference diagrams in issues and PRs
4. **Share**: Use diagrams in presentations and papers

---

**Last Updated**: November 7, 2025
**Version**: Phase 0 Complete
**Contributing**: Add new diagrams to this file as system evolves!
