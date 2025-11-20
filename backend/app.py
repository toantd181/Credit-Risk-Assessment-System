from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Neural Network Model (same as your training code)
class ImprovedCreditRiskNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rates=[0.5, 0.4, 0.3, 0.2]):
        super(ImprovedCreditRiskNN, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout_rates[0])
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout_rates[1])
        
        # Layer 3
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.dropout3 = nn.Dropout(dropout_rates[2])
        
        # Layer 4
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.bn4 = nn.BatchNorm1d(hidden_dims[3])
        self.dropout4 = nn.Dropout(dropout_rates[3])
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dims[3], 1)
    
    def forward(self, x):
        x = self.input_bn(x)
        
        # Layer 1
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Layer 2
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Layer 3
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Layer 4
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        # Output
        x = torch.sigmoid(self.fc_out(x))
        
        return x

# Global variables for model and preprocessing
model = None
scaler = None
feature_columns = None
label_encoders = None

def load_model():
    """Load the trained model and preprocessing objects"""
    global model, scaler, feature_columns, label_encoders
    
    try:
        # FIXED: Set weights_only=False to allow loading scikit-learn objects
        checkpoint = torch.load(
            'credit_risk_pytorch_model_v2.pth', 
            map_location=torch.device('cpu'),
            weights_only=False  # This is the critical fix
        )
        
        # Extract components
        scaler = checkpoint['scaler']
        feature_columns = checkpoint['feature_columns']
        label_encoders = checkpoint['label_encoders']
        
        # Initialize model
        input_dim = len(feature_columns)
        model = ImprovedCreditRiskNN(input_dim=input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"   • Input features: {input_dim}")
        print(f"   • Feature columns: {len(feature_columns)}")
        print(f"\n📋 Expected Features:")
        for i, col in enumerate(feature_columns, 1):
            print(f"   {i}. {col}")
        print("")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def prepare_input(data):
    """Prepare input data for prediction"""
    try:
        # Create feature vector matching training format
        features = {}
        
        # Direct numeric features
        features['Credit_Score'] = data.get('credit_score', 650)
        features['income'] = data.get('income', 50000)
        features['loan_amount'] = data.get('loan_amount', 25000)
        features['LTV'] = data.get('ltv', 80)
        features['age_numeric'] = data.get('age', 35)
        
        # Calculate engineered features
        features['debt_to_income'] = features['loan_amount'] / (features['income'] + 1)
        
        # Debug: print what features we're using
        print(f"\n📊 Input Features:")
        print(f"   Credit Score: {features['Credit_Score']}")
        print(f"   Income: ${features['income']:,.0f}")
        print(f"   Loan Amount: ${features['loan_amount']:,.0f}")
        print(f"   LTV: {features['LTV']}%")
        print(f"   Age: {features['age_numeric']}")
        print(f"   Debt-to-Income: {features['debt_to_income']:.4f}")
        
        # Create feature array in correct order
        feature_array = []
        for col in feature_columns:
            if col in features:
                feature_array.append(features[col])
            elif col.endswith('_encoded'):
                # Handle encoded categorical features
                orig_col = col.replace('_encoded', '')
                if orig_col in label_encoders and orig_col in data:
                    try:
                        encoded_val = label_encoders[orig_col].transform([data[orig_col]])[0]
                        feature_array.append(encoded_val)
                    except:
                        feature_array.append(0)
                else:
                    feature_array.append(0)
            else:
                # Default value for missing features
                feature_array.append(0)
        
        print(f"   Total features prepared: {len(feature_array)}")
        print(f"   Expected features: {len(feature_columns)}")
        
        return np.array(feature_array, dtype=np.float32).reshape(1, -1)
    
    except Exception as e:
        print(f"Error preparing input: {str(e)}")
        return None

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Credit Risk Assessment API',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict credit risk for loan application"""
    try:
        # Get input data
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare features
        features = prepare_input(data)
        
        if features is None:
            return jsonify({'error': 'Invalid input data'}), 400
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled)
            model_output = model(features_tensor).item()
            risk_prob = 1 - model_output
        
        # DEBUG: Print raw prediction
        print(f"\n🔍 DEBUG INFO:")
        print(f"   Raw model output: {risk_prob}")
        print(f"   Feature vector shape: {features_scaled.shape}")
        print(f"   Number of features: {len(feature_columns)}")
        
        # Risk probability (0 = high risk/default, 1 = low risk/no default)
        # Convert to risk score (higher = better)
        risk_score = risk_prob * 100
        
        # Determine decision based on thresholds
        APPROVE_THRESHOLD = 70
        REVIEW_THRESHOLD = 40
        
        if risk_score >= APPROVE_THRESHOLD:
            decision = 'APPROVED'
            risk_level = 'LOW'
            recommendation = '✅ Low risk profile - Approve with standard terms'
        elif risk_score >= REVIEW_THRESHOLD:
            decision = 'MANUAL_REVIEW'
            risk_level = 'MEDIUM'
            recommendation = '⚠️ Moderate risk - Manual underwriting review required'
        else:
            decision = 'REJECTED'
            risk_level = 'HIGH'
            recommendation = '❌ High risk profile - Does not meet minimum requirements'
        
        # Analyze key factors
        factors = analyze_factors(data, risk_score)
        
        # Build response
        response = {
            'success': True,
            'decision': decision,
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'confidence': round(abs(risk_prob - 0.5) * 2 * 100, 2),
            'factors': factors,
            'application_summary': {
                'credit_score': data.get('credit_score', 'N/A'),
                'income': data.get('income', 'N/A'),
                'loan_amount': data.get('loan_amount', 'N/A'),
                'ltv': data.get('ltv', 'N/A'),
                'dti': data.get('dti', 'N/A'),
                'age': data.get('age', 'N/A')
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def analyze_factors(data, risk_score):
    """Analyze key risk factors"""
    factors = []
    
    credit_score = data.get('credit_score', 650)
    ltv = data.get('ltv', 80)
    dti = data.get('dti', 35)
    income = data.get('income', 50000)
    loan_amount = data.get('loan_amount', 25000)
    employment_length = data.get('employment_length', 0)
    
    # Credit Score Analysis
    if credit_score >= 740:
        factors.append({
            'type': 'positive',
            'label': '🟢 Excellent Credit Score',
            'detail': f'Credit score of {credit_score} is excellent (740+)'
        })
    elif credit_score >= 670:
        factors.append({
            'type': 'positive',
            'label': '🟢 Good Credit Score',
            'detail': f'Credit score of {credit_score} meets requirements'
        })
    elif credit_score >= 580:
        factors.append({
            'type': 'warning',
            'label': '🟡 Fair Credit Score',
            'detail': f'Credit score of {credit_score} is below recommended threshold'
        })
    else:
        factors.append({
            'type': 'negative',
            'label': '🔴 Poor Credit Score',
            'detail': f'Credit score of {credit_score} is significantly below requirements'
        })
    
    # DTI Analysis
    if dti < 30:
        factors.append({
            'type': 'positive',
            'label': '🟢 Excellent DTI',
            'detail': f'DTI of {dti:.1f}% shows strong financial capacity'
        })
    elif dti < 43:
        factors.append({
            'type': 'positive',
            'label': '🟢 Acceptable DTI',
            'detail': f'DTI of {dti:.1f}% is within acceptable range'
        })
    else:
        factors.append({
            'type': 'negative',
            'label': '🔴 High DTI',
            'detail': f'DTI of {dti:.1f}% exceeds safe threshold (43%)'
        })
    
    # LTV Analysis
    if ltv < 80:
        factors.append({
            'type': 'positive',
            'label': '🟢 Low LTV',
            'detail': f'LTV of {ltv}% shows good equity cushion'
        })
    elif ltv < 90:
        factors.append({
            'type': 'warning',
            'label': '🟡 Moderate LTV',
            'detail': f'LTV of {ltv}% is acceptable but higher risk'
        })
    else:
        factors.append({
            'type': 'negative',
            'label': '🔴 High LTV',
            'detail': f'LTV of {ltv}% indicates high leverage risk'
        })
    
    # Employment Stability
    if employment_length >= 5:
        factors.append({
            'type': 'positive',
            'label': '🟢 Strong Employment History',
            'detail': f'{employment_length} years shows excellent stability'
        })
    elif employment_length >= 2:
        factors.append({
            'type': 'positive',
            'label': '🟢 Stable Employment',
            'detail': f'{employment_length} years shows adequate stability'
        })
    else:
        factors.append({
            'type': 'warning',
            'label': '🟡 Limited Employment History',
            'detail': f'{employment_length} years is relatively short'
        })
    
    # Loan to Income Ratio
    loan_to_income = loan_amount / (income + 1)
    if loan_to_income < 2:
        factors.append({
            'type': 'positive',
            'label': '🟢 Conservative Loan Amount',
            'detail': f'Loan is {loan_to_income:.1f}x annual income'
        })
    elif loan_to_income < 4:
        factors.append({
            'type': 'warning',
            'label': '🟡 Moderate Loan Amount',
            'detail': f'Loan is {loan_to_income:.1f}x annual income'
        })
    else:
        factors.append({
            'type': 'negative',
            'label': '🔴 High Loan Amount',
            'detail': f'Loan is {loan_to_income:.1f}x annual income'
        })
    
    return factors

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_count': len(feature_columns) if feature_columns else 0,
        'encoders_loaded': label_encoders is not None
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 CREDIT RISK ASSESSMENT API")
    print("="*70)
    
    # Load model
    if load_model():
        print("\n✅ Starting server...")
        print("📡 API will be available at: http://localhost:5000")
        print("="*70 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model. Please check if 'credit_risk_pytorch_model_v2.pth' exists.")