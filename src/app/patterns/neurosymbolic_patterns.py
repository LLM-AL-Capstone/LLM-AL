import json
import os
from pathlib import Path
from jinja2 import Template
from ..llm.ollama import OllamaClient
from ..utils.io import write_json
import datetime

class NeuroSymbolicPatternLearner:
    def __init__(self, cfg):
        self.client = OllamaClient(cfg["model_gen"], temperature=0.1)
        self.cfg = cfg
        self.patterns_cache = {}
        
    def get_patterns_filename(self, task_name, model_name):
        """Generate consistent filename for patterns"""
        clean_model = model_name.replace(':', '_').replace('/', '_')
        return f"patterns_{task_name}_{clean_model}.json"
    
    def load_existing_patterns(self, task_name, model_name):
        """Load patterns from cache if they exist"""
        patterns_dir = Path("reports/patterns")
        patterns_dir.mkdir(parents=True, exist_ok=True)
        
        patterns_file = patterns_dir / self.get_patterns_filename(task_name, model_name)
        
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                print(f"📁 Loaded existing patterns from: {patterns_file}")
                return patterns_data.get("patterns", {})
            except Exception as e:
                print(f"⚠️  Error loading patterns: {e}")
        
        return None
    
    def save_patterns(self, patterns, task_name, model_name, train_size):
        """Save patterns with metadata"""
        patterns_dir = Path("reports/patterns")
        patterns_dir.mkdir(parents=True, exist_ok=True)
        
        patterns_file = patterns_dir / self.get_patterns_filename(task_name, model_name)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        patterns_data = {
            "task": task_name,
            "model": model_name,
            "train_size": train_size,
            "timestamp": timestamp,
            "patterns": patterns
        }
        
        write_json(str(patterns_file), patterns_data)
        print(f"💾 Saved patterns to: {patterns_file}")
        
        # Also create a symlink to latest
        latest_link = patterns_dir / f"patterns_{task_name}_latest.json"
        try:
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(patterns_file.name)
        except Exception as e:
            print(f"Note: Could not create symlink: {e}")
        
        return str(patterns_file)
    
    def learn_patterns_for_label(self, examples, label, all_labels):
        """Learn symbolic patterns for a specific label using Programming-by-Example"""
        
        # Sample examples to avoid context overflow
        sample_examples = examples[:8] if len(examples) > 8 else examples
        other_labels = [l for l in all_labels if l != label]
        
        # Load and render the centralized pattern learning prompt
        prompt_path = Path("prompts/patterns/pattern_learning.txt")
        try:
            with open(prompt_path, 'r') as f:
                template = Template(f.read())
            
            # Convert examples to the expected format
            example_dicts = [{"text": ex} for ex in sample_examples]
            
            prompt = template.render(
                label=label,
                examples=example_dicts,
                other_labels=other_labels
            )
        except Exception as e:
            print(f"    Error loading pattern template: {e}")
            # Fallback to inline prompt with English enforcement
            prompt = f"""You are an English-speaking linguistic analyzer specializing in pattern recognition. Respond ONLY in English.

TASK: Analyze examples labeled as "{label}" and learn symbolic patterns using Programming-by-Example approach.

EXAMPLES FOR LABEL "{label}":
{chr(10).join([f"- \"{ex}\"" for ex in sample_examples])}

OTHER LABELS: {', '.join(other_labels)}

PATTERN LANGUAGE COMPONENTS:
- POS tags: NOUN, VERB, ADJ, ADV, etc.
- Entity types: $LOCATION, $DATE, $ORG, $PERSON  
- Word stems: [word] (e.g., [have] matches has, had, having)
- Soft match: (expensive) matches synonyms like pricey, costly
- Wildcards: * for any token sequence
- Exact phrases: "must match exactly"

INSTRUCTIONS:
1. Infer 3-5 symbolic patterns that characterize "{label}" examples
2. Focus on syntactic structure AND semantic content 
3. Identify which parts can be MODIFIED to change the label
4. Patterns should distinguish "{label}" from other labels
5. Write ALL descriptions in clear English only

OUTPUT (JSON only, all text in English):
{{
    "patterns": [
        {{
            "rule": "symbolic pattern using components above",
            "description": "what this pattern captures (in English)",
            "modifiable_segments": ["segment1", "segment2"],
            "confidence": 0.85
        }}
    ]
}}

Respond with JSON only. Use English for all text fields."""

        try:
            response = self.client.run(prompt, system="You are an English-speaking assistant. Always respond in English only.", max_tokens=512, retries=1)
            return self._parse_patterns(response)
        except Exception as e:
            print(f"    Error learning patterns for {label}: {e}")
            return []
    
    def identify_candidate_phrases(self, text, patterns, original_label):
        """Identify candidate phrases from learned patterns"""
        
        if not patterns or original_label not in patterns:
            return {
                "phrases": ["semantic content"],
                "pattern_rule": "general structure",
                "strategy": "general"
            }
        
        label_patterns = patterns[original_label]
        if not label_patterns:
            return {
                "phrases": ["semantic content"], 
                "pattern_rule": "general structure",
                "strategy": "general"
            }
        
        # Use highest confidence pattern
        best_pattern = max(label_patterns, key=lambda p: p.get("confidence", 0))
        
        return {
            "phrases": best_pattern.get("modifiable_segments", ["semantic content"]),
            "pattern_rule": best_pattern.get("rule", "general structure"),
            "strategy": "pattern_guided",
            "description": best_pattern.get("description", "")
        }
    
    def learn_all_patterns(self, train_df, text_field, label_field, labels, task_name, force_relearn=False):
        """Learn patterns for all labels with caching"""
        
        model_name = self.cfg["model_gen"]
        
        # Try to load existing patterns first
        if not force_relearn:
            existing_patterns = self.load_existing_patterns(task_name, model_name)
            if existing_patterns:
                print("🎯 Using cached patterns. Use --force-relearn to regenerate.")
                return existing_patterns
        
        print("🧩 Step 3: Learning Neuro-Symbolic Patterns...")
        print("=" * 50)
        
        all_patterns = {}
        
        for label in labels:
            print(f"  Learning patterns for: {label}")
            
            label_examples = train_df[train_df[label_field] == label][text_field].tolist()
            
            if len(label_examples) < 2:
                print(f"    Insufficient examples ({len(label_examples)}), using general patterns")
                all_patterns[label] = []
                continue
            
            print(f"    Analyzing {len(label_examples)} examples...")
            patterns = self.learn_patterns_for_label(label_examples, label, labels)
            all_patterns[label] = patterns
            
            print(f"    Learned {len(patterns)} patterns:")
            for i, pattern in enumerate(patterns, 1):
                rule = pattern.get('rule', 'N/A')
                conf = pattern.get('confidence', 0)
                print(f"      {i}. {rule} (conf: {conf:.2f})")
        
        # Save the learned patterns
        self.save_patterns(all_patterns, task_name, model_name, len(train_df))
        
        return all_patterns
    
    def _parse_patterns(self, response):
        """Parse pattern learning response"""
        try:
            # Handle markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end != -1:
                    response = response[start:end].strip()
            
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                pattern_data = json.loads(json_str)
                return pattern_data.get("patterns", [])
            else:
                return []
                
        except json.JSONDecodeError:
            return []