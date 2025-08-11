class ClinicalOphthalmologyLLM:
    def __init__(self):
        self.conditions = [
            'diabetic_retinopathy', 'macular_edema', 'amd',
            'retinal_detachment', 'increased_cup_disc', 'other'
        ]
        self.recommendations_db = {
            'diabetic_retinopathy': 'Optimize diabetes control and refer to ophthalmology.',
            'macular_edema': 'Consider OCT and anti-VEGF evaluation.',
            'amd': 'Monitor drusen, consider AREDS and retina clinic follow-up.',
            'retinal_detachment': 'Urgent retina consult recommended.',
            'increased_cup_disc': 'Assess IOP and glaucoma risk; schedule evaluation.'
        }

    def format_findings(self, probs, thr=0.5):
        F = []
        urgent = []
        for cond, p in zip(self.conditions, probs):
            if p >= thr:
                conf = "high" if p >= 0.8 else "moderate" if p >= 0.65 else "low"
                F.append(f"- {cond.replace('_',' ').title()}: {conf} ({p:.2f})")
                if cond in {"retinal_detachment"} and p >= 0.5:
                    urgent.append(cond)
        if not F:
            F = ["- No strong abnormality detected (all below threshold)"]
        return F, urgent

    def summarize(self, patient, probs):
        findings, urgent = self.format_findings(probs)
        recs = []

        if urgent:
            recs.append("🚨 URGENT: Findings may require immediate retina evaluation.")
            for u in urgent:
                if u in self.recommendations_db:
                    recs.append(f"• {self.recommendations_db[u]}")

        # Generic
        if not recs:
            recs.append("• Routine follow-up and risk-factor optimization as indicated.")

        age = patient.get("age", "unknown")
        if isinstance(age, (int, float)) and age >= 60:
            recs.append("• Age ≥60: consider regular retinal screening.")

        return (
            "📋 OPHTHALMOLOGY AI SUMMARY\n\n"
            f"Patient: age={patient.get('age','?')}, sex={patient.get('sex','?')}\n\n"
            "RETINAL FINDINGS:\n" + "\n".join(findings) + "\n\n" +
            "RECOMMENDATIONS:\n" + "\n".join(recs) + "\n\n" +
            "⚠️ This AI output requires clinical validation."
        )

def create_clinical_llm():
    return ClinicalOphthalmologyLLM()
