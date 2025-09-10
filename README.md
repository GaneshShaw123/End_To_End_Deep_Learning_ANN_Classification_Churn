# ANN-CLassification-Churn

CSV File: Churn_Modelling.csv
           │
           ▼
   ┌─────────────────┐
   │ Load Dataset    │  ← pandas.read_csv()
   └─────────────────┘
           │
           ▼
   ┌─────────────────────────────┐
   │ Drop Irrelevant Columns     │  ← RowNumber, CustomerId, Surname
   └─────────────────────────────┘
           │
           ▼
   ┌─────────────────────────────┐
   │ Encode Categorical Features │
   │ - Gender → LabelEncoder     │
   │ - Geography → OneHotEncoder │
   └─────────────────────────────┘
           │
           ▼
   ┌─────────────────────────────┐
   │ Split Features & Target     │
   │ X = All columns except Exited│
   │ y = Exited                  │
   └─────────────────────────────┘
           │
           ▼
   ┌─────────────────────────────┐
   │ Train/Test Split            │  ← 80% train, 20% test
   └─────────────────────────────┘
           │
           ▼
   ┌─────────────────────────────┐
   │ Scale Features              │  ← StandardScaler
   └─────────────────────────────┘
           │
           ▼
   ┌─────────────────────────────┐
   │ Neural Network Model        │
   │ - Input Layer               │
   │ - Hidden Layer 1 (64 neurons) │
   │ - Hidden Layer 2 (32 neurons) │
   │ - Output Layer (1 neuron, sigmoid) │
   └─────────────────────────────┘
           │
           ▼
   ┌─────────────────────────────┐
   │ Compile Model               │
   │ - Optimizer: Adam           │
   │ - Loss: BinaryCrossentropy  │
   │ - Metrics: Accuracy         │
   └─────────────────────────────┘
           │
           ▼
   ┌─────────────────────────────┐
   │ Training Model              │
   │ - Validation: Test set      │
   │ - Epochs: up to 100        │
   │ - EarlyStopping & TensorBoard │
   └─────────────────────────────┘
           │
           ▼
   ┌─────────────────────────────┐
   │ Save Artifacts              │
   │ - model.h5                  │
   │ - scaler.pkl                │
   │ - label_encoder_gender.pkl  │
   │ - onehot_encoder_geo.pkl    │
   └─────────────────────────────┘
           │
           ▼
   ┌─────────────────────────────┐
   │ Deployment / Prediction     │
   │ - Load saved model/scalers  │
   │ - Input new data → predict  │
   └─────────────────────────────┘
