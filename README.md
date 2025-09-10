# ANN-CLassification-Churn

## Churn Prediction Pipeline

```mermaid
flowchart TD
    A[CSV File: Churn_Modelling.csv] --> B[Load Dataset (pandas)]
    B --> C[Drop Irrelevant Columns: RowNumber, CustomerId, Surname]
    C --> D[Encode Categorical Features]
    D --> D1[Gender → LabelEncoder]
    D --> D2[Geography → OneHotEncoder]
    D1 --> E[Split Features & Target]
    D2 --> E
    E --> F[Train/Test Split: 80% train, 20% test]
    F --> G[Scale Features: StandardScaler]
    G --> H[Neural Network Model]
    H --> H1[Input Layer]
    H --> H2[Hidden Layer 1 (64 neurons, ReLU)]
    H --> H3[Hidden Layer 2 (32 neurons, ReLU)]
    H --> H4[Output Layer (1 neuron, Sigmoid)]
    H4 --> I[Compile Model: Adam, BinaryCrossentropy, Accuracy]
    I --> J[Train Model: validation on test set, EarlyStopping, TensorBoard]
    J --> K[Save Artifacts: model.h5, scaler.pkl, label_encoder_gender.pkl, onehot_encoder_geo.pkl]
    K --> L[Deployment / Prediction: Load saved objects, predict churn on new data]

