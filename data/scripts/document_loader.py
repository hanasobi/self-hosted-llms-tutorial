"""
document_loader.py - Document Metadata Definition

Defines the metadata structure for AWS documentation files.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DocumentMetadata:
    """
    Structured metadata for a single document.
    
    Read from the JSON file in the S3 directory.
    Falls back to sensible defaults if fields are missing.
    """
    service: str  # e.g. "EC2", "S3", "IAM"
    doc_type: str  # e.g. "FAQ", "UserGuide", "DeveloperGuide"
    source_url: Optional[str] = None
    last_updated: Optional[str] = None
    
    @classmethod
    def from_json(cls, json_data: Dict, default_service: str = "Unknown") -> 'DocumentMetadata':
        """
        Create DocumentMetadata from a JSON dict.
        
        Robust against missing fields — we don't want a single missing
        field to crash the entire processing pipeline.
        
        Args:
            json_data: Dictionary from metadata.json
            default_service: Fallback service name if not in JSON
            
        Returns:
            DocumentMetadata instance
        """
        return cls(
            service=json_data.get('service', default_service),
            doc_type=json_data.get('doc_type', 'Unknown'),
            source_url=json_data.get('source_url'),
            last_updated=json_data.get('last_updated')
        )
