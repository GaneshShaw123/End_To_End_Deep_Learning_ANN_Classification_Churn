# ANN-CLassification-Churn

## End-to-End Pipeline: Training + Prediction

```mermaid
flowchart TD
    %% Training Pipeline
    A1[CSV File Churn Modelling CSV] --> B1[Load Dataset with pandas]
    B1 --> C1[Drop Columns RowNumber CustomerId Surname]
    C1 --> D1[Encode Categorical Features]
    D1 --> D11[Gender using LabelEncoder]
    D1 --> D12[Geography using OneHotEncoder]
    D11 --> E1[Split Features and Target]
    D12 --> E1
    E1 --> F1[Train Test Split 80-20]
    F1 --> G1[Scale Features using StandardScaler]
    G1 --> H1[Neural Network Model]
    H1 --> H11[Input Layer]
    H1 --> H12[Hidden Layer 1 - 64 neurons ReLU]
    H1 --> H13[Hidden Layer 2 - 32 neurons ReLU]
    H1 --> H14[Output Layer - 1 neuron Sigmoid]
    H14 --> I1[Compile Model - Adam Optimizer BinaryCrossentropy Accuracy]
    I1 --> J1[Train Model with Validation EarlyStopping TensorBoard]
    J1 --> K1[Save Artifacts model.h5 scaler.pkl encoders.pkl]

    %% Prediction Pipeline
    L1[Input Customer Data] --> M1[Convert to DataFrame]
    M1 --> N1[Encode Gender using LabelEncoder]
    M1 --> O1[One-Hot Encode Geography using OneHotEncoder]
    N1 --> P1[Drop original Geography column and concat encoded columns]
    O1 --> P1
    P1 --> Q1[Scale Features using StandardScaler]
    Q1 --> R1[Load Trained Model model.h5]
    R1 --> S1[Predict Churn Probability]
    S1 --> T1{Is probability greater than 0.5?}
    T1 -->|Yes| U1[Customer likely to churn]
    T1 -->|No| V1[Customer not likely to churn]

    %% Connect Training to Prediction
    K1 --> R1
