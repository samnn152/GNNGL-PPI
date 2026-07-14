# Feature-First MVC

The repository is organized by feature first (`train`, `test`, `common`) and
by MVC responsibility inside each feature.

```text
src/
├── train/
│   ├── controllers/    CLI input, defaults, pipeline composition, run command
│   ├── models/         config, context, pipeline, trainer, steps, result types
│   └── views/          terminal observer and progress rendering
├── test/
│   ├── controllers/    evaluation CLI and input defaults
│   └── models/         evaluation config, evaluator, analysis toolkit
└── common/
    ├── models/         PPI graph, GNNGL-PPI, fusion, local encoders, losses
    └── data/           datasets, parsing, graph extraction, reporting, MASSA
```

## MVC dependency flow

```mermaid
flowchart LR
    TrainInput[CLI arguments] --> TrainController[train.controllers]
    TrainController --> TrainModels[train.models]
    TrainController --> TrainViews[train.views]
    TrainModels --> SharedModels[common.models]
    TrainModels --> SharedData[common.data]
    TrainModels -. observer events .-> TrainViews

    TestInput[Evaluation CLI] --> TestController[test.controllers]
    TestController --> TestModels[test.models]
    TestModels --> SharedModels
    TestModels --> SharedData
```

## Training sequence

```mermaid
sequenceDiagram
    participant CLI as main.py
    participant C as TrainController
    participant P as TrainingSetupPipeline
    participant T as GNNTrainer
    participant V as TerminalTrainUI

    CLI->>C: run_training()
    C->>C: parse arguments and resolve defaults
    C->>P: build and run setup steps
    P-->>C: prepared TrainContext
    C->>V: create observer
    C->>T: create session and options
    T-->>V: lifecycle and metric events
    T-->>C: TrainResult
    C->>C: append result CSV
```

The Model owns training state and behavior. The Controller translates CLI
input into model operations and composes the workflow. The View never starts
training; it only renders observer events emitted by the model.
