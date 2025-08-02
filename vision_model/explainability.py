"""
Phase 3: Explainability + Safety
Grad-CAM, attention visualization, and safety mechanisms
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAMExplainer:
    """Grad-CAM implementation for vision model explainability"""
    
    def __init__(self, model, target_layer_name='backbone'):
        self.model = model
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # Find target layer
        for name, module in model.named_modules():
            if target_layer_name in name:
                self.target_layer = module
                break
        
        if self.target_layer is None:
            print(f"⚠️ Layer {target_layer_name} not found, using last conv layer")
            # Find last convolutional layer
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    self.target_layer = module
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[1, 2])  # [C]
        
        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:])  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.detach().cpu().numpy()
    
    def visualize_cam(self, original_image, cam, alpha=0.4):
        """Overlay CAM on original image"""
        # Resize CAM to match image
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        if len(original_image.shape) == 3:
            overlay = alpha * heatmap + (1 - alpha) * original_image
        else:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            overlay = alpha * heatmap + (1 - alpha) * original_rgb
        
        return overlay.astype(np.uint8)

class SafetyValidator:
    """Safety validation for clinical AI predictions"""
    
    def __init__(self, confidence_threshold=0.7, uncertainty_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # Define high-risk conditions requiring immediate attention
        self.high_risk_conditions = [
            'retinal_detachment', 'diabetic_retinopathy', 'macular_edema'
        ]
        
        # Define condition severity mapping
        self.condition_severity = {
            'retinal_detachment': 'EMERGENCY',
            'diabetic_retinopathy': 'HIGH',
            'macular_edema': 'HIGH',
            'amd': 'MODERATE',
            'hypertensive_retinopathy': 'MODERATE',
            'other': 'LOW'
        }
    
    def validate_predictions(self, predictions, condition_names):
        """Validate AI predictions for safety"""
        validation_results = {
            'safe_to_proceed': True,
            'requires_specialist': False,
            'emergency_referral': False,
            'uncertainty_flags': [],
            'high_confidence_predictions': [],
            'recommendations': []
        }
        
        for i, (pred, condition) in enumerate(zip(predictions, condition_names)):
            # Check confidence levels
            if pred > self.confidence_threshold:
                validation_results['high_confidence_predictions'].append({
                    'condition': condition,
                    'confidence': float(pred),
                    'severity': self.condition_severity.get(condition, 'UNKNOWN')
                })
                
                # Check for high-risk conditions
                if condition in self.high_risk_conditions:
                    if condition == 'retinal_detachment':
                        validation_results['emergency_referral'] = True
                        validation_results['recommendations'].append(
                            f"🚨 EMERGENCY: {condition} detected with {pred:.2f} confidence - Immediate ophthalmology referral required"
                        )
                    else:
                        validation_results['requires_specialist'] = True
                        validation_results['recommendations'].append(
                            f"⚡ HIGH PRIORITY: {condition} detected with {pred:.2f} confidence - Urgent specialist consultation recommended"
                        )
            
            # Check for uncertainty
            elif 0.3 < pred < 0.7:
                validation_results['uncertainty_flags'].append({
                    'condition': condition,
                    'confidence': float(pred),
                    'reason': 'Moderate uncertainty - consider additional imaging or specialist review'
                })
        
        # Overall safety assessment
        if validation_results['emergency_referral']:
            validation_results['safe_to_proceed'] = False
            validation_results['recommendations'].insert(0, "🚨 EMERGENCY PROTOCOL: Immediate specialist referral required")
        elif validation_results['requires_specialist']:
            validation_results['recommendations'].insert(0, "⚡ SPECIALIST REFERRAL: High-risk findings detected")
        elif len(validation_results['uncertainty_flags']) > 3:
            validation_results['recommendations'].append("🔍 ADDITIONAL REVIEW: Multiple uncertain findings - consider repeat imaging")
        
        return validation_results
    
    def generate_safety_report(self, predictions, condition_names, patient_data):
        """Generate comprehensive safety report"""
        validation = self.validate_predictions(predictions, condition_names)
        
        # Risk stratification based on patient factors
        age = patient_data.get('age', 0)
        diabetes_duration = patient_data.get('diabetes_time', 0)
        
        risk_factors = []
        if age > 65:
            risk_factors.append("Advanced age (>65)")
        if diabetes_duration > 10:
            risk_factors.append("Long-standing diabetes (>10 years)")
        if patient_data.get('insulin_use'):
            risk_factors.append("Insulin-dependent diabetes")
        
        safety_report = f"""
🛡️ AI SAFETY VALIDATION REPORT

RISK LEVEL: {'🚨 EMERGENCY' if validation['emergency_referral'] else '⚡ HIGH' if validation['requires_specialist'] else '✅ ROUTINE'}

HIGH-CONFIDENCE FINDINGS:
{chr(10).join([f"• {finding['condition']}: {finding['confidence']:.2f} ({finding['severity']} severity)" for finding in validation['high_confidence_predictions']]) if validation['high_confidence_predictions'] else '• No high-confidence findings'}

UNCERTAINTY FLAGS:
{chr(10).join([f"• {flag['condition']}: {flag['confidence']:.2f} - {flag['reason']}" for flag in validation['uncertainty_flags']]) if validation['uncertainty_flags'] else '• No significant uncertainties'}

PATIENT RISK FACTORS:
{chr(10).join([f"• {factor}" for factor in risk_factors]) if risk_factors else '• No additional risk factors identified'}

CLINICAL RECOMMENDATIONS:
{chr(10).join([f"• {rec}" for rec in validation['recommendations']]) if validation['recommendations'] else '• Routine follow-up as clinically indicated'}

⚠️ DISCLAIMER: This AI assessment requires validation by qualified medical professional.
"""
        
        return safety_report, validation

print("✅ Phase 3: Explainability + Safety components created")
