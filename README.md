# ANN-CLassification-Churn

## Churn Prediction Pipeline

```mermaid
flowchart TD
    A[CSV File Churn Modelling CSV] --> B[Load Dataset with pandas]
    B --> C[Drop Columns RowNumber CustomerId Surname]
    C --> D[Encode Categorical Features]
    D --> D1[Gender using LabelEncoder]
    D --> D2[Geography using OneHotEncoder]
    D1 --> E[Split Features and Target]
    D2 --> E
    E --> F[Train Test Split 80-20]
    F --> G[Scale Features using StandardScaler]
    G --> H[Neural Network Model]
    H --> H1[Input Layer]
    H --> H2[Hidden Layer 1 64 neurons ReLU]
    H --> H3[Hidden Layer 2 32 neurons ReLU]
    H --> H4[Output Layer 1 neuron Sigmoid]
    H4 --> I[Compile Model Adam Optimizer BinaryCrossentropy Accuracy]
    I --> J[Train Model with Validation EarlyStopping TensorBoard]
    J --> K[Save Artifacts model.h5 scaler.pkl encoders.pkl]
    K --> L[Deployment Load Objects Predict Churn]
