import json

class ClinicalOphthalmologyLLM:
    def __init__(self):
        self.conditions = [
            'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
            'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 
            'hemorrhage', 'retinal_detachment', 'myopic_fundus', 
            'increased_cup_disc', 'other'
        ]
        
        # Clinical templates
        self.recommendations_db = {
            'diabetic_retinopathy': 'Urgent diabetes management and ophthalmology referral',
            'macular_edema': 'Consider anti-VEGF therapy evaluation',
            'amd': 'AMD monitoring and AREDS supplementation',
            'high_risk': 'Immediate ophthalmology consultation required'
        }
    
    def format_patient_context(self, patient_data):
        return {
            'age': patient_data.get('age', 'unknown'),
            'sex': 'male' if patient_data.get('sex') == 1 else 'female',
            'diabetes_time': patient_data.get('diabetes_time', 'unknown'),
            'insulin_use': 'yes' if patient_data.get('insulin_use') == 1 else 'no'
        }
    
    def format_vision_findings(self, predictions, confidence_threshold=0.5):
        findings = []
        high_risk_conditions = []
        
        for i, (condition, prob) in enumerate(zip(self.conditions, predictions)):
            if prob > confidence_threshold:
                confidence_level = 'high' if prob > 0.8 else 'moderate' if prob > 0.6 else 'low'
                findings.append(f"- {condition.replace('_', ' ').title()}: {confidence_level} confidence ({prob:.2f})")
                
                if prob > 0.7 and condition in ['diabetic_retinopathy', 'macular_edema', 'retinal_detachment']:
                    high_risk_conditions.append(condition)
        
        if not findings:
            findings = ["- No significant abnormalities detected"]
        
        return findings, high_risk_conditions
    
    def generate_recommendations(self, patient_context, high_risk_conditions, findings):
        recommendations = []
        
        # Risk-based recommendations
        if high_risk_conditions:
            recommendations.append("🚨 URGENT: High-risk findings detected")
            for condition in high_risk_conditions:
                if condition in self.recommendations_db:
                    recommendations.append(f"• {self.recommendations_db[condition]}")
        
        # Patient-specific recommendations
        if patient_context['diabetes_time'] != 'unknown' and int(patient_context['diabetes_time']) > 5:
            recommendations.append("• Enhanced diabetes monitoring recommended")
        
        if patient_context['age'] != 'unknown' and int(patient_context['age']) > 60:
            recommendations.append("• Regular retinal screening due to age risk factor")
        
        if not recommendations:
            recommendations.append("• Routine follow-up as clinically indicated")
        
        return recommendations
    
    def generate_clinical_report(self, patient_data, vision_predictions):
        patient_context = self.format_patient_context(patient_data)
        findings, high_risk_conditions = self.format_vision_findings(vision_predictions)
        recommendations = self.generate_recommendations(patient_context, high_risk_conditions, findings)
        
        report = f"""
📋 OPHTHALMOLOGY AI ASSESSMENT REPORT

PATIENT INFORMATION:
- Age: {patient_context['age']} years
- Sex: {patient_context['sex']}
- Diabetes duration: {patient_context['diabetes_time']} years  
- Insulin use: {patient_context['insulin_use']}

RETINAL FINDINGS:
{chr(10).join(findings)}

CLINICAL RECOMMENDATIONS:
{chr(10).join(recommendations)}

⚠️  Note: This AI assessment requires clinical validation by qualified ophthalmologist.
"""
        
        return {
            'report': report,
            'findings': findings,
            'recommendations': recommendations,
            'high_risk': len(high_risk_conditions) > 0
        }

def create_clinical_llm():
    return ClinicalOphthalmologyLLM()

print("✅ Clinical LLM created")
