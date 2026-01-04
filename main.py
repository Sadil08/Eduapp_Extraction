from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from model_loader import ModelLoader
from token_logger import token_logger
from PIL import Image
import io
import os

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

def get_extraction_prompt(subject_name: Optional[str] = None, lesson_name: Optional[str] = None, doc_type: str = "question"):
    """
    Get context-primed extraction prompt
    
    Args:
        subject_name: Subject name (e.g., "Mathematics")
        lesson_name: Lesson name (e.g., "Trigonometry", "Calculus")
        doc_type: Type of document - "question", "modelanswer", or "handwritten"
    """
    context_prefix = ""
    
    # Build context information
    if subject_name or lesson_name:
        context_parts = []
        if subject_name:
            context_parts.append(f"Subject: {subject_name}")
        if lesson_name:
            context_parts.append(f"Lesson: {lesson_name}")
        context_prefix = f"Context: {', '.join(context_parts)}\n\n"
    
    # Get base subject-specific prompt
    base_prompt = ""
    if subject_name and subject_name in SUBJECT_PROMPTS:
        base_prompt = SUBJECT_PROMPTS[subject_name]
    else:
        base_prompt = """Extract absolutely all text, tables, and mathematical formulas from this image.
Output mathematical formulas in LaTeX format. Do not add any conversational text, just the extracted content."""
    
    # Add handwritten-specific instructions
    if doc_type == "handwritten":
        handwritten_instructions = """
**CRITICAL - Handwritten Student Answer:**
- Transcribe EXACTLY what is written
- **DO NOT correct mathematical errors or logical mistakes**
- If handwriting is unclear, infer the most likely characters based on the context above
- Preserve the student's original work and intent, including any mistakes
- This is for grading purposes - we need the student's actual answer, not a corrected version"""
        return context_prefix + handwritten_instructions + "\n\n" + base_prompt
    
    # For questions and model answers, just use context + base prompt
    return context_prefix + base_prompt



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

class BatchExtractionItem(BaseModel):
    id: str
    extracted_text: str

class BatchExtractionResponse(BaseModel):
    results: List[BatchExtractionItem]
    total_processed: int

@app.on_event("startup")
async def startup_event():
    # Preload model on startup to avoid delay on first request
    # model_loader.load_model() # Commented out for dev speed, uncomment for prod
    pass

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "EduApp AI Service"}

@app.post("/extract", response_model=ExtractionResponse)
async def extract_text(
    file: UploadFile = File(...), 
    subject: Optional[str] = Form(default=None),
    lesson: Optional[str] = Form(default=None),
    docType: Optional[str] = Form(default="question")
):
    try:
        # DEBUG: Log received parameters with VISIBLE formatting
        print("=" * 60)
        print("[EXTRACTION DEBUG] === RECEIVED PARAMS ===")
        print(f"  subject: '{subject}' (type: {type(subject).__name__})")
        print(f"  lesson:  '{lesson}' (type: {type(lesson).__name__})")
        print(f"  docType: '{docType}' (type: {type(docType).__name__})")
        print("=" * 60)
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Use context-primed prompt for better extraction
        prompt = get_extraction_prompt(subject, lesson, docType if docType else "question")
        
        # DEBUG: Log the FULL generated prompt
        print("[EXTRACTION DEBUG] === FULL PROMPT ===")
        print(prompt)
        print("=" * 60)
        
        extracted, usage = model_loader.predict(image, prompt)
        
        # Log token usage
        token_logger.log_usage(
            operation="extract",
            input_tokens=usage['prompt_token_count'],
            output_tokens=usage['candidates_token_count'],
            doc_type=docType or "question",
            subject=subject,
            lesson=lesson,
            image_included=usage['image_included'],
            batch_size=1
        )
        
        return ExtractionResponse(extracted_text=extracted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-batch", response_model=BatchExtractionResponse)
async def extract_text_batch(
    files: List[UploadFile] = File(...),
    ids: str = Form(...),  # Comma-separated IDs matching each file
    contexts: Optional[str] = Form(None),  # JSON mapping id -> {subject, lesson, docType}
    subject: Optional[str] = Form(None),  # Fallback if contexts not provided
    lesson: Optional[str] = Form(None),  # Fallback if contexts not provided
    docType: Optional[str] = Form(default="question")  # Fallback if contexts not provided
):
    """
    Batch extract text from multiple images in a single request with per-image context.
    
    - files: List of image files to process
    - ids: Comma-separated list of IDs (e.g., "q1,q2,q3") matching each file
    - contexts: Optional JSON string mapping each ID to its context:
        Example: '{"q1": {"subject": "Mathematics", "lesson": "Trigonometry", "docType": "question"},
                   "q2": {"subject": "Physics", "lesson": "Mechanics", "docType": "question"}}'
    - subject, lesson, docType: Fallback values if contexts not provided (all images use same context)
    
    This reduces API overhead by processing multiple images together.
    """
    try:
        import json
        
        id_list = [id.strip() for id in ids.split(",")]
        
        if len(files) != len(id_list):
            raise HTTPException(
                status_code=400, 
                detail=f"Mismatch: {len(files)} files but {len(id_list)} IDs provided"
            )
        
        if len(files) > 20:
            raise HTTPException(
                status_code=400,
                detail="Maximum 20 images per batch request"
            )
        
        # Parse contexts if provided
        context_map = {}
        if contexts:
            try:
                context_map = json.loads(contexts)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in contexts parameter: {str(e)}"
                )
        
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for file, item_id in zip(files, id_list):
            try:
                # Get context for this specific image
                if item_id in context_map:
                    ctx = context_map[item_id]
                    img_subject = ctx.get("subject", subject)
                    img_lesson = ctx.get("lesson", lesson)
                    img_docType = ctx.get("docType", docType if docType else "question")
                else:
                    # Use fallback values
                    img_subject = subject
                    img_lesson = lesson
                    img_docType = docType if docType else "question"
                
                # Generate prompt for this specific image
                prompt = get_extraction_prompt(img_subject, img_lesson, img_docType)
                
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                extracted, usage = model_loader.predict(image, prompt)
                
                # Aggregate token usage
                total_input_tokens += usage['prompt_token_count']
                total_output_tokens += usage['candidates_token_count']
                
                results.append(BatchExtractionItem(
                    id=item_id,
                    extracted_text=extracted
                ))
            except Exception as e:
                # Add error result but continue processing other images
                results.append(BatchExtractionItem(
                    id=item_id,
                    extracted_text=f"[ERROR: {str(e)}]"
                ))
        
        # Log aggregated batch usage
        token_logger.log_usage(
            operation="extract_batch",
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            doc_type=docType or "question",
            subject=subject,  # Use fallback subject for logging
            lesson=lesson,    # Use fallback lesson for logging
            image_included=True,
            batch_size=len(results)
        )
        
        return BatchExtractionResponse(
            results=results,
            total_processed=len(results)
        )
    except HTTPException:
        raise
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
        
        response_text, usage = model_loader.predict(dummy_image, prompt)
        
        # Log token usage for marking
        token_logger.log_usage(
            operation="mark",
            input_tokens=usage['prompt_token_count'],
            output_tokens=usage['candidates_token_count'],
            doc_type="marking",
            subject=None,
            lesson=None,
            image_included=False,
            batch_size=1
        )
        
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
