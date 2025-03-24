from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Set
import re
from .base import ProbingDataset, ProbingExample
from .position_finder import PositionFinder

@dataclass
class TemplateVariable:
    """Represents a variable in a template."""
    name: str
    values: List[str]
    metadata: Optional[Dict] = None  # e.g., sentiment polarity, POS tag, etc.
    class_bound: bool = False  # Whether this variable must match classes with other variables
    class_key: Optional[str] = None  # Metadata key that defines the class (e.g., "sentiment")

@dataclass
class Template:
    """Represents a template with variables."""
    template: str  # e.g., "I thought this movie was {ADJ}, I {VERB} it."
    variables: Dict[str, TemplateVariable]  # e.g., {"ADJ": TemplateVariable(...)}
    metadata: Optional[Dict] = None
    
    def get_marker(self, var_name: str) -> str:
        """Get the marker for a variable in the template."""
        return f"{{{var_name}}}"
    
    def get_all_markers(self) -> Dict[str, str]:
        """Get all variable markers in the template."""
        return {name: self.get_marker(name) for name in self.variables.keys()}
    
    def validate(self) -> bool:
        """Validate that all variables in template exist in variables dict."""
        template_vars = set(re.findall(r'\{([^}]+)\}', self.template))
        defined_vars = set(self.variables.keys())
        return template_vars == defined_vars

class TemplatedDataset:
    """Dataset based on templates with variables."""
    
    def __init__(
        self,
        templates: List[Template],
        metadata: Optional[Dict] = None
    ):
        """Initialize TemplatedDataset.
        
        Args:
            templates: List of templates to use
            metadata: Additional metadata about the dataset
        """
        self.templates = templates
        self.metadata = metadata or {}
        
        # Validate all templates
        for template in templates:
            if not template.validate():
                raise ValueError(f"Invalid template: {template.template}")
    
    def to_probing_dataset(
        self,
        label_from_metadata: Optional[str] = None,  # e.g., "sentiment"
        label_map: Optional[Dict[str, int]] = None,
        auto_add_positions: bool = True
    ) -> ProbingDataset:
        """Convert to ProbingDataset.
        
        Args:
            label_from_metadata: Key in variable metadata to use as label
            label_map: Mapping from metadata values to numeric labels
            auto_add_positions: Whether to automatically add positions for variables
        """
        examples = []
        
        # Generate all combinations for each template
        for template in self.templates:
            # Separate class-bound and neutral variables
            bound_vars = []
            neutral_vars = []
            for name, var in template.variables.items():
                if var.class_bound:
                    bound_vars.append(name)
                else:
                    neutral_vars.append(name)
            
            # Get all possible classes from the first bound variable
            possible_classes = set()
            if bound_vars:
                first_var = template.variables[bound_vars[0]]
                if first_var.metadata and first_var.class_key:
                    possible_classes = set(first_var.metadata[first_var.class_key])
            
            # For each class, generate combinations
            for class_value in possible_classes or [None]:
                # Get class-consistent values for bound variables
                bound_values = []
                for var_name in bound_vars:
                    var = template.variables[var_name]
                    if class_value is not None:
                        # Filter values to match the current class
                        class_indices = [
                            i for i, v in enumerate(var.metadata[var.class_key])
                            if v == class_value
                        ]
                        values = [var.values[i] for i in class_indices]
                    else:
                        values = var.values
                    bound_values.append(values)
                
                # Get values for neutral variables
                neutral_values = [
                    template.variables[name].values 
                    for name in neutral_vars
                ]
                
                # Generate combinations
                from itertools import product
                bound_combinations = list(product(*bound_values)) if bound_values else [()] 
                neutral_combinations = list(product(*neutral_values)) if neutral_values else [()]
                
                for bound_combo in bound_combinations:
                    for neutral_combo in neutral_combinations:
                        # Create text by substituting values
                        text = template.template
                        var_metadata: Dict[str, Optional[Dict]] = {}
                        
                        # Insert bound variables
                        for name, value in zip(bound_vars, bound_combo):
                            marker = template.get_marker(name)
                            text = text.replace(marker, value)

                            # UPDATED: Slice out the relevant metadata
                            var = template.variables[name]
                            sliced_md = {}
                            if var.metadata:
                                # For each key in var.metadata, pick the sub-value that matches this 'value'
                                # based on its position in var.values
                                value_index = var.values.index(value)
                                sliced_md = {
                                    k: var.metadata[k][value_index]
                                    for k in var.metadata
                                    if isinstance(var.metadata[k], list) and len(var.metadata[k]) > value_index
                                }
                            var_metadata[name] = sliced_md if sliced_md else var.metadata
                        
                        # Insert neutral variables
                        for name, value in zip(neutral_vars, neutral_combo):
                            marker = template.get_marker(name)
                            text = text.replace(marker, value)

                            # UPDATED: Only store relevant slice for neutral variables as well
                            var = template.variables[name]
                            sliced_md = {}
                            if var.metadata:
                                value_index = var.values.index(value)
                                sliced_md = {
                                    k: var.metadata[k][value_index]
                                    for k in var.metadata
                                    if isinstance(var.metadata[k], list) and len(var.metadata[k]) > value_index
                                }
                            var_metadata[name] = sliced_md if sliced_md else var.metadata
                        
                        # Determine label if specified
                        label = 0
                        label_text = ""
                        if label_from_metadata and label_map and class_value is not None:
                            label = label_map[class_value]
                            label_text = class_value
                        
                        # Create example
                        example = ProbingExample(
                            text=text,
                            label=label,
                            label_text=label_text,
                            metadata={
                                "template": template.template,
                                "variables": var_metadata,
                                "class": class_value,
                                **(template.metadata if template.metadata else {})
                            }
                        )
                        examples.append(example)
        
        # Create dataset
        dataset = ProbingDataset(
            examples=examples,
            metadata=self.metadata
        )
        
        # Add positions for each variable if requested
        if auto_add_positions:
            for template in self.templates:
                for var_name in template.variables:
                    finder = PositionFinder.from_template(
                        template=template.template,
                        marker=template.get_marker(var_name)
                    )
                    dataset.add_target_positions(key=var_name, finder=finder)
        
        return dataset
    
    @classmethod
    def from_movie_sentiment_template(
        cls,
        adjectives: Dict[str, List[str]],  # e.g., {"positive": [...], "negative": [...]}
        verbs: Dict[str, List[str]]  # e.g., {"positive": [...], "negative": [...]}
    ) -> "TemplatedDataset":
        """Create dataset from movie sentiment template.
        
        Args:
            adjectives: Dictionary mapping sentiment to list of adjectives
            verbs: Dictionary mapping sentiment to list of verbs
        """
        # Create variables
        adj_var = TemplateVariable(
            name="ADJ",
            values=[adj for lst in adjectives.values() for adj in lst],
            metadata={"sentiment": [k for k, v in adjectives.items() for _ in v]},
            class_bound=True,
            class_key="sentiment"
        )
        
        verb_var = TemplateVariable(
            name="VERB",
            values=[verb for lst in verbs.values() for verb in lst],
            metadata={"sentiment": [k for k, v in verbs.items() for _ in v]},
            class_bound=True,
            class_key="sentiment"
        )
        
        # Create template with exact spacing
        template = Template(
            template="I thought this movie was {ADJ}, I {VERB} it.",
            variables={
                "ADJ": adj_var,
                "VERB": verb_var
            },
            metadata={"task": "sentiment_classification"}
        )
        
        return cls(templates=[template])
    
    @classmethod 
    def from_mood_story_template(
        cls,
        names: List[str],
        verbs: Dict[str, List[str]]  # e.g., {"positive": [...], "negative": [...]}
    ) -> "TemplatedDataset":
        """Create dataset from mood story template."""
        # Create variables
        name_var = TemplateVariable(
            name="NAME",
            values=names
        )
        
        verb_var = TemplateVariable(
            name="VERB",
            values=[v for lst in verbs.values() for v in lst],
            metadata={"sentiment": [k for k, v in verbs.items() for _ in v]}
        )
        
        # Create template
        template = Template(
            template="{NAME} {VERB} parties, and does so whenever possible.",
            variables={
                "NAME": name_var,
                "VERB": verb_var
            },
            metadata={"task": "mood_prediction"}
        )
        
        return cls(templates=[template])