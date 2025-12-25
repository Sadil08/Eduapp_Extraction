from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from model_loader import ModelLoader
from PIL import Image
import io

# Subject-specific extraction prompts for better accuracy
SUBJECT_PROMPTS = {
    "Mathematics": """Extract all text and mathematical expressions from this image.
- Use LaTeX for formulas: \\frac{3}{4}, x^2, \\int, \\sum
- Preserve equation structure exactly as shown
- Include all questions, working, and answers
- Common patterns: algebraic equations, calculus, geometry, statistics""",
    
    "Physics": """Extract all text, formulas, and diagrams from this image.
- Use LaTeX for physics notation: F=ma, v=\\frac{d}{t}, E=mc^2
- Note any graphs, force diagrams, or circuit diagrams
- Units are critical: include all units (m/s, kg, N, J)
- Include all questions and numerical values""",
    
    "Chemistry": """Extract all text and chemical content from this image.
- Chemical formulas: H₂O, CH₃COOH (use subscripts/superscripts)
- Reactions: use → for reaction arrows
- Include molecular structures, equations, and lab procedure steps
- Note any diagrams or tables""",
    
    "Biology": """Extract all text and diagrams from this image.
- Preserve labeled diagrams, tables, and classifications
- Use italics for Latin/scientific names
- Include process flows, cycles, and hierarchies
- Extract all questions and descriptive content"""
}

def get_extraction_prompt(subject_name: Optional[str] = None, base_instruction: str = ""):
    """Get subject-specific extraction prompt or generic fallback"""
    if subject_name and subject_name in SUBJECT_PROMPTS:
        return SUBJECT_PROMPTS[subject_name] + (f"\n\n{base_instruction}" if base_instruction else "")
    
    # Generic fallback
    return f"""Extract absolutely all text, tables, and mathematical formulas from this image, including any questions asked.
Output mathematical formulas in LaTeX format. Do not add any conversational text, just the extracted content.
{base_instruction}"""


app = FastAPI(title="EduApp AI Service")

# Initialize model loader
model_loader = ModelLoader.get_instance()

class MarkingRequest(BaseModel):
    question_text: str
    model_answer_text: Optional[str] = None
    student_answer_text: str
    total_marks: int

class ExtractionResponse(BaseModel):
    extracted_text: str

class MarkingResponse(BaseModel):
    feedback: str
    marks_awarded: int
    lessons_to_review: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    # Preload model on startup to avoid delay on first request
    # model_loader.load_model() # Commented out for dev speed, uncomment for prod
    pass

@app.post("/extract", response_model=ExtractionResponse)
async def extract_text(file: UploadFile = File(...), subject: Optional[str] = None):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Use subject-specific prompt for better extraction
        prompt = get_extraction_prompt(subject)
        
        extracted = model_loader.predict(image, prompt)
        
        return ExtractionResponse(extracted_text=extracted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mark", response_model=MarkingResponse)
async def mark_answer(request: MarkingRequest):
    try:
        # Construct a prompt for the model to act as a strict examiner
        prompt = f"""
        You are an expert academic examiner. Grade the student's answer based on the question and the model answer (if provided).
        
        Question: {request.question_text}
        
        Model Answer: {request.model_answer_text if request.model_answer_text else 'N/A'}
        
        Student Answer: {request.student_answer_text}
        
        Total Marks Available: {request.total_marks}
        
        Task:
        1. Compare the student's answer with the model answer/question requirements.
        2. Assign a mark out of {request.total_marks}.
        3. Provide constructive feedback.
        4. Suggest lessons or topics to review if the answer is incorrect.
        
        Output Format (JSON):
        {{
            "marks": <int>,
            "feedback": "<string>",
            "lessons_to_review": "<string>"
        }}
        """
        
        # We use the same model for text generation logic, passing a dummy image or using a text-only method if supported.
        # Qwen2.5-VL allows text-only input if we just don't pass 'image'.
        # However, our loader is built for VL. We might need a dummy image or a separate method.
        # For simplicity in this iteration, we'll assume we might pass a blank image or handle text-only.
        # Let's adjust ModelLoader to handle text-only if image is None.
        
        # Update: We will treat this as a pure text generation task for now, assuming the model supports it.
        # If Qwen VL *requires* an image, we can generate a 1x1 white pixel image.
        
        dummy_image = Image.new('RGB', (10, 10), color='white')
        
        response_text = model_loader.predict(dummy_image, prompt)
        
        # Parse JSON from response (basic parsing for now, assuming model behaves)
        # In production, use structured output parsing or regex
        import json
        import re
        
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return MarkingResponse(
                marks_awarded=data.get("marks", 0),
                feedback=data.get("feedback", "No feedback provided"),
                lessons_to_review=data.get("lessons_to_review", "")
            )
        else:
            # Fallback if specific formatting failed
            return MarkingResponse(
                marks_awarded=0,
                feedback=response_text,
                lessons_to_review=""
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
