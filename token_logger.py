import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Gemini 1.5 Flash Pricing (per 1M tokens)
INPUT_COST_PER_MILLION = 0.075  # $0.075 per 1M input tokens
OUTPUT_COST_PER_MILLION = 0.30   # $0.30 per 1M output tokens

# Log directory
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

class TokenLogger:
    """Utility class for logging token usage and calculating costs"""
    
    def __init__(self):
        self.log_file = self._get_current_log_file()
        self._ensure_log_file_exists()
    
    def _get_current_log_file(self) -> Path:
        """Get log file path for current month"""
        now = datetime.now()
        filename = f"token_usage_{now.year}_{now.month:02d}.csv"
        return LOG_DIR / filename
    
    def _ensure_log_file_exists(self):
        """Create log file with headers if it doesn't exist"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'operation',
                    'doc_type',
                    'subject',
                    'lesson',
                    'input_tokens',
                    'output_tokens',
                    'total_tokens',
                    'estimated_cost_usd',
                    'image_included',
                    'batch_size'
                ])
    
    @staticmethod
    def calculate_cost(input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost in USD based on token usage"""
        input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
        return input_cost + output_cost
    
    def log_usage(
        self,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        doc_type: str = "unknown",
        subject: Optional[str] = None,
        lesson: Optional[str] = None,
        image_included: bool = False,
        batch_size: int = 1
    ):
        """
        Log token usage to CSV file
        
        Args:
            operation: Type of operation (extract, extract_batch, mark)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            doc_type: Type of document (question, modelanswer, handwritten)
            subject: Subject name
            lesson: Lesson name
            image_included: Whether image was included in request
            batch_size: Number of items in batch (1 for single)
        """
        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)
        
        # Ensure we're writing to current month's file
        current_log_file = self._get_current_log_file()
        if current_log_file != self.log_file:
            self.log_file = current_log_file
            self._ensure_log_file_exists()
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                operation,
                doc_type,
                subject or '',
                lesson or '',
                input_tokens,
                output_tokens,
                total_tokens,
                f"{cost:.6f}",
                image_included,
                batch_size
            ])
        
        # Also log summary to console
        print(f"[TOKEN USAGE] {operation} | Tokens: {total_tokens} (in: {input_tokens}, out: {output_tokens}) | Cost: ${cost:.6f}")
    
    def get_summary(self, days: int = 30) -> dict:
        """Get summary statistics for the last N days"""
        # This could be extended to parse the log files and generate statistics
        # For now, just return a placeholder
        return {
            "message": f"Log file: {self.log_file}",
            "note": "Use Excel or pandas to analyze the CSV log file"
        }

# Global instance
token_logger = TokenLogger()
