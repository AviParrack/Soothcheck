"""
Conversation formatting utilities for NTML datasets.

Handles formatting sampled statements into conversational format with precise
position tracking for statement-level labeling.
"""

import re
from typing import List, Dict, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConversationFormatter:
    """Formats statements into conversational format with position tracking."""
    
    def __init__(self):
        """Initialize formatter."""
        self.numbered_list_pattern = re.compile(r'^\d+\.\s*', re.MULTILINE)
    
    def format_conversation(
        self, 
        system_prompt: str, 
        statements: List[Dict], 
        conversation_id: str,
        add_timestamp: bool = True
    ) -> Dict[str, Any]:
        """Format statements into conversational JSONL format.
        
        Args:
            system_prompt: System message content
            statements: List of statement dictionaries with 'statement' and 'label' keys
            conversation_id: Unique identifier for this conversation
            add_timestamp: Whether to add timestamp to output
            
        Returns:
            Dictionary in JSONL format with messages and labels
        """
        # Create assistant response with numbered list
        assistant_content, statement_positions = self._create_numbered_response(statements)
        
        # Build messages array
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
        
        # Extract labels and create statement-level metadata
        ground_truth = [bool(stmt['label']) for stmt in statements]
        statement_level = self._create_statement_metadata(statements, statement_positions)
        
        # Count truths and lies
        truth_count = sum(ground_truth)
        lie_count = len(ground_truth) - truth_count
        
        # Build complete conversation record
        conversation = {
            "id": conversation_id,
            "messages": messages,
            "labels": {
                "ratio": f"{truth_count}T{lie_count}L",
                "truth_count": truth_count,
                "lie_count": lie_count,
                "ground_truth": ground_truth,
                "statement_level": statement_level
            }
        }
        
        # Add timestamp if requested
        if add_timestamp:
            conversation["created_at"] = datetime.utcnow().isoformat() + "Z"
        
        return conversation
    
    def _create_numbered_response(self, statements: List[Dict]) -> Tuple[str, List[Tuple[int, int]]]:
        """Create numbered list response with position tracking.
        
        Args:
            statements: List of statement dictionaries
            
        Returns:
            Tuple of (formatted_response, position_list)
            where position_list contains (char_start, char_end) for each statement
        """
        lines = []
        positions = []
        current_pos = 0
        
        for i, stmt in enumerate(statements, 1):
            statement_text = stmt['statement'].strip()
            
            # Create numbered line
            line = f"{i}. {statement_text}"
            
            # Calculate positions for the statement text (excluding number and period)
            number_prefix = f"{i}. "
            statement_start = current_pos + len(number_prefix)
            statement_end = statement_start + len(statement_text)
            
            positions.append((statement_start, statement_end))
            lines.append(line)
            
            # Update position for next line (including newline if not last)
            current_pos += len(line)
            if i < len(statements):
                current_pos += 1  # for newline character
        
        response = "\n".join(lines)
        return response, positions
    
    def _create_statement_metadata(
        self, 
        statements: List[Dict], 
        positions: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """Create detailed metadata for each statement.
        
        Args:
            statements: List of statement dictionaries
            positions: List of (char_start, char_end) tuples
            
        Returns:
            List of statement metadata dictionaries
        """
        statement_level = []
        
        for i, (stmt, (char_start, char_end)) in enumerate(zip(statements, positions)):
            metadata = {
                "text": stmt['statement'].strip(),
                "is_true": bool(stmt['label']),
                "position": i,
                "char_start": char_start,
                "char_end": char_end
            }
            
            # Add source information if available
            if 'source' in stmt:
                metadata['source'] = stmt['source']
            
            # Add statement ID if available
            if 'statement_id' in stmt:
                metadata['statement_id'] = stmt['statement_id']
            
            statement_level.append(metadata)
        
        return statement_level
    
    def validate_positions(self, conversation: Dict[str, Any]) -> bool:
        """Validate that character positions are correct.
        
        Args:
            conversation: Formatted conversation dictionary
            
        Returns:
            True if positions are valid, False otherwise
        """
        try:
            assistant_content = conversation['messages'][1]['content']
            statement_level = conversation['labels']['statement_level']
            
            for stmt_meta in statement_level:
                char_start = stmt_meta['char_start']
                char_end = stmt_meta['char_end']
                expected_text = stmt_meta['text']
                
                # Extract text at specified position
                actual_text = assistant_content[char_start:char_end]
                
                if actual_text != expected_text:
                    logger.error(f"Position mismatch: expected '{expected_text}', got '{actual_text}'")
                    return False
            
            return True
            
        except (KeyError, IndexError) as e:
            logger.error(f"Position validation failed: {e}")
            return False
    
    def extract_statements_from_response(self, response: str) -> List[str]:
        """Extract individual statements from a numbered response.
        
        Args:
            response: Assistant response with numbered list
            
        Returns:
            List of statement texts (without numbers)
        """
        lines = response.split('\n')
        statements = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            if not line:
                continue
            
            # Remove number prefix (e.g., "1. ", "2. ", etc.)
            match = re.match(r'^\d+\.\s*(.+)$', line)
            if match:
                statement_text = match.group(1)
                statements.append(statement_text)
            else:
                # Line doesn't match expected format, but include it anyway
                logger.warning(f"Unexpected line format: {line}")
                statements.append(line)
        
        return statements


def format_conversation(
    system_prompt: str, 
    statements: List[Dict], 
    conversation_id: str,
    add_timestamp: bool = True
) -> Dict[str, Any]:
    """Convenience function to format a conversation.
    
    Args:
        system_prompt: System message content
        statements: List of statement dictionaries
        conversation_id: Unique identifier for this conversation
        add_timestamp: Whether to add timestamp
        
    Returns:
        Formatted conversation dictionary
    """
    formatter = ConversationFormatter()
    return formatter.format_conversation(system_prompt, statements, conversation_id, add_timestamp) 